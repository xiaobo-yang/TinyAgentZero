import re
import requests
import json
from typing import List
import torch

from .utils import normalize_answer


class SearchEnv:

    paraphrase = False 
    # if paraphrase is True, the env will return reflection cot on response_str, 
    # then these cot will be used as new thinking cot in agent


    VALID_ACTIONS = ['search', 'answer']

    def __init__(self, n_envs, search_url, topk, is_validation=False):
        self.n_envs = n_envs
        self.topk = topk
        self.search_url = search_url
        self.is_validation = is_validation

        self._success = {idx: False for idx in range(self.n_envs)}
        self.reward = {idx: [] for idx in range(self.n_envs)}
        self.trajs = {
            'question': {idx: None for idx in range(self.n_envs)},
            'trajectory': {idx: [] for idx in range(self.n_envs)},
            }

    def run(self, responses_str: List[str], batch, chat_template=None):
        """
        receive model response, run the environment
        """
        # abort text after </search> or </answer>
        filtered_responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        actions, queries = self.extract_actions(filtered_responses_str)
        search_queries = [query for action, query in zip(actions, queries) if action == 'search']
        search_results = self.batch_search(search_queries)
        
        feedbacks, dones = [], []
        for idx, (action, query, data_item) in enumerate(zip(actions, queries, batch)):
            if self.trajs['question'][idx] is None:
                golden_answers = data_item.non_tensor_batch['extra_info']['golden_answers']
                # normalize golden answers
                golden_answers = [normalize_answer(gans) for gans in golden_answers]
                self.trajs['question'][idx] = {
                    'question': data_item.non_tensor_batch['extra_info']['question'],
                    'golden answers': golden_answers,
                }
            else:
                golden_answers = self.trajs['question'][idx]['golden answers']

            
            feedback = ""
            

            if self.finished(idx):
                done = True
                reward = 0.0
            else:
                if action not in self.VALID_ACTIONS:
                    observation = (
                        'Your previous action is invalid. If you want to search, '  # TODO: add chat template
                        'you should put the query between <search> and </search>. '
                        'If you want to give the final answer, you should put the answer '
                        'between <answer> and </answer>. Please try again.'
                    )
                    if chat_template is not None:
                        observation = (
                            f"\n{chat_template['start']}{chat_template['user']}\n{observation}{chat_template['end']}\n"
                            f"{chat_template['start']}{chat_template['assistant']}\n{chat_template['think_start']}"
                        )
                        end_pattern = f"{chat_template['end']}"
                        if end_pattern not in filtered_responses_str[idx]:
                            observation = f"{chat_template['end']}" + observation
                    done = False
                    reward = 0.0
                elif action == 'answer':
                    # normalize model answer
                    model_answer = normalize_answer(query)
                    # if self.is_validation:
                    #     observation = ''
                    #     done = True
                    #     if model_answer in golden_answers: # exact match(em)
                    #         reward = 1.0
                    #     else:
                    #         reward = 0.0
                    # else:
                    if model_answer in golden_answers:
                        observation = ''
                        done = True
                        reward = 1.0
                    else:
                        observation = (
                            'Your answer is incorrect! You need to try another answer. '
                            'If you find that the answer is not included in the search results, '
                            'you should search again.'
                        )
                        if chat_template is not None:
                            observation = (
                                f"\n{chat_template['start']}{chat_template['user']}\n{observation}{chat_template['end']}\n"
                                f"{chat_template['start']}{chat_template['assistant']}\n{chat_template['think_start']}"
                            )
                            end_pattern = f"{chat_template['end']}"
                            if end_pattern not in filtered_responses_str[idx]:
                                observation = f"{chat_template['end']}" + observation
                        done = False
                        reward = 0.0
                elif action == 'search':
                    observation = f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n'
                    done = False
                    reward = 0.0
                
                if self.paraphrase:
                    if action == 'search':
                        observation = (
                            f'Please review the search results and determine if they contain relevant information to the question. If relevant information is found, highlight it and explain how it can be used to formulate an answer. If not, indicate that the search results are insufficient and a new search is needed.'
                            f'Search results are: {observation}\n'
                        )
                    else:
                        observation = (
                            f'Regarding your answer, the system feedback is: "{observation}"\n'
                            f'Analyze any errors or limitations in your response. Consider strategies to improve your approach and successfully complete this task.'
                        )

                feedback += observation

            self.reward[idx].append(reward)
            feedbacks.append(feedback)
            dones.append(done)
            
            if done:
                self._success[idx] = True

            self.trajs['trajectory'][idx].append({
                'current_step': len(self.reward[idx]),
                'done': done,
                'action': action,
                'query': query,
                'step_reward': reward,
                'filtered response': filtered_responses_str[idx],
                'feedback': feedback,
            })

        step_rewards = [r[-1] for r in self.reward.values()]
        
        # debug
        import random
        ridx = random.randint(0, len(self.reward)-1)
        print(f"--------------------------------")
        print(f"Question: {batch[ridx].non_tensor_batch['extra_info']['question']}")
        print(f"Golden answers: {[normalize_answer(gans) for gans in batch[ridx].non_tensor_batch['extra_info']['golden_answers']]}")
        print(f"Action: {actions[ridx]}")
        print(f"Query: {queries[ridx]}")
        print(f"Filtered response: {filtered_responses_str[ridx]}")


        return filtered_responses_str, feedbacks, dones, step_rewards, self.trajs
    
    def extract_actions(self, responses_str: List[str]):
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        queries = []
                
        for response in responses_str:
            pattern = r'<(search|answer)>(.*?)</\1>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                action = match.group(1)
                query = match.group(2).strip()  # Return only the content inside the tags
            else:
                action = None
                query = ''
            
            actions.append(action)
            queries.append(query)
            
        return actions, queries
    
    def finished(self, idx):
        return self._success[idx]

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True
        }
        
        return requests.post(self.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

        
