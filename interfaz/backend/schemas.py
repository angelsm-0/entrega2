from pydantic import BaseModel

class UserInput(BaseModel):
   edad: int
   altura: float
   peso: float
   faf: int
   ch2o: int
   fcvc: int
