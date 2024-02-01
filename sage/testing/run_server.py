from sage.utils.common import CONSOLE
from sage.utils.trigger_server import AllServerRunner

if __name__ == "__main__":
    server_runner = AllServerRunner()
    server_runner.run()
    CONSOLE.print("ran server!!")
