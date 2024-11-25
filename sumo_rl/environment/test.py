from typing import Callable
class test:
    def __init__(self,name) -> None:
        self._name_ = name
        print(self._name_)
    
    @classmethod
    def register(cls, fn: Callable):
        print(fn())
        
a = test('qqq')
def hello() -> int:
    return 1
