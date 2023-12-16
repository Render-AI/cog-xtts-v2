from cog import BasePredictor, File
import io

def train(param: str) -> File:
    return io.StringIO("hello " + param)