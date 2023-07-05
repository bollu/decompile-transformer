#!/usr/bin/env python3
import jax
from jax import numpy as jnp
from jax import random 
from jax.typing import *
from jax import lax
from jax import grad, jit, vmap
from jax import random
# TODO: find nice way to perform gradient accumulation?
from jax.tree_util import tree_map
# import torch # use pytorch for now.
from typing import *
from dataclasses import dataclass, is_dataclass

def ceil_div(num : int, denom : int) -> int:
    return (num + denom - 1) // denom

def grouper(iterable, n): 
    out = []
    for _ in range(n):
        try:
            out.append(next(iterable))
        except StopIteration:
            break
    return out

def grouper_iter(iterable, n): 
    """
    group iterable into groups of size n.
    Last group may be at most n
    """
    while True:
        out = []
        for _ in range(n):
            try:
                out.append(next(iterable))
            except StopIteration:
                yield out 
                return # done yielding
        yield out

print(grouper([1,2,3,4,5], 2))


@dataclass
class Config:
    nepochs : int 
    nbatches : int # number of batches per SGD step. 
    ministep_size : int # size of ministep in each batch
    
class Model:
    params : List[ArrayLike] # all model parameters to be gradient descended on.
    def __init__(self, Config):  pass
    
class Optimizer:
    def __init__(self): pass
        


DataClass = NewType('DataClass', Any)


class DataCollator:
    def __call__(self, data : List[DataClass]) -> DataClass:
        pass

class Dataset:
    def mk_iterator(self) -> Iterator[DataClass]:
        pass

class Evaluator:
    pass


@dataclass
class TrainingState:
    model : Model 
    batch_loss : torch.tensor
    epoch_loss : float 
    bix : int 
    eix : int 
    data_iter : Iterator[DataClass]
    pass


@dataclass
class TrainingCommand:
    stop : bool = False 
    save : bool = False 



class TrainingCallback:
    def before_train(self, config : Config, state : TrainingState) -> TrainingCommand:
        return TrainingCommand()
    def before_batch(self, config : Config, state : TrainingState) -> TrainingCommand:
        return TrainingCommand()
    def before_mini_step(self, config : Config, state : TrainingState) -> TrainingCommand:
        return TrainingCommand()
    def after_mini_step(self, config : Config, state : TrainingState) -> TrainingCommand:
        return TrainingCommand()
    def after_batch(self) -> TrainingCommand:
        return TrainingCommand()
    def before_epoch(self, config : Config, state : TrainingState) -> TrainingCommand:
        return TrainingCommand()
    def after_epoch(self) -> TrainingCommand:
        return TrainingCommand()

class TrainingLoop:
    _config : Config 
    _model : Model 
    _dataset : Dataset
    _collator : DataCollator
    _evaluator : Evaluator
    _optimizer : Optimizer
    _state : TrainingState
    
    """Users should freely modify callback to add new callbacks"""
    callbacks : List[TrainingCallback]
    def __init__(self, config : Config, model : Model, dataset : Dataset,
                 collator: DataCollator, evaluator : Evaluator,
                 optimizer: Optimizer):
        self._config = config 
        self._state = TrainingState(model=model,
                                    batch_loss=0.0,
                                    epoch_loss=0.0,
                                    data_iter=dataset.mk_iterator())
        self._model = model 
        self._dataset = dataset
        self._collator = collator
        self._evaluator = evaluator
        self._optimizer = optimizer
        self.callbacks = []
        pass

        
    def run(self):
        for eix in range(self.config.nepochs):
            self._state.eix = eix
            for cb in self.callbacks:
                cb.before_epoch(self, self._state)
            self._state.epoch_loss = 0
            self.run_epoch(self)
            for cb in self.callbacks:
                cb.after_epoch(self, self._state)

    def run_epoch(self):
        for bix in range(self.state.nbatches):
            self._state.bix = bix
            for cb in self.callbacks:
                cb.before_batch(self, self._state)
            self.run_batch(self)

    def run_batch(self):
        size_per_batch = ceil_div(len(self._dataset), self._config.nbatches)
        nministeps = ceil_div(size_per_batch, self._config.ministep_size)
        self._state.batch_loss = torch.tensor(0.0, dtype=float)

        for cb in self.callbacks:
            cb.before_batch(self, self._state)

        for ms in range(self.nministeps):
            self.run_ministep(self, ms)
        
        self._optimizer.optimize(self._state.batch_grad)
        for cb in self.callbacks:
            cb.after_batch(self, self._state)

    def run_ministep(self, msix : int):
        # collate data for ministep
        data = self._collator(grouper(self._state.data_iter, self._config.ministep_size))
        ms_loss = self._model.forward(data)
        self._state.batch_loss += ms_loss