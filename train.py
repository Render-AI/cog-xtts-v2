from cog import BasePredictor, File
import io

def train(out_path: Input('out_path', default="/tmp")) -> File:
    return io.StringIO("hello " + out_path)