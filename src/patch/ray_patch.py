import os
import ray
from unittest.mock import patch
from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    _unwrap_ray_remote,
    _bind_workers_method_to_parent,
)


def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function should return a class instance that delegates the calls to every 
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = None
    for key, cls in class_dict.items():
        if worker_cls == None:
            worker_cls = cls.cls.__ray_actor_class__.__base__.__base__  # NOTE: need double __base__ for patched worker class
        else:
            assert worker_cls == cls.cls.__ray_actor_class__.__base__.__base__, \
                'the worker class should be the same when share the same process'
        cls_dict[key] = cls.cls
        init_args_dict[key] = {'args': cls.args, 'kwargs': cls.kwargs}

    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    class WorkerDict(worker_cls):

        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                # directly instantiate the class without remote
                # in worker class, e.g. <verl.single_controller.base.worker.Worker> when DISABLE_WORKER_INIT == 1 it will return immediately
                with patch.dict(os.environ, {'DISABLE_WORKER_INIT': '1'}):
                    self.worker_dict[key] = user_defined_cls(*init_args_dict[key].get('args', ()),
                                                             **init_args_dict[key].get('kwargs', {}))

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    return remote_cls