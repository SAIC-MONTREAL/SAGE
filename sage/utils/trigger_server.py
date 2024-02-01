"""
AIOHTTP servers used for triggering the LLMs.
The currently supported triggers are:
- condition (llm can register conditions which will trigger it)
- manual post to the server
"""
import asyncio
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
import time
from typing import Callable, Optional, Any


import aiohttp
from aiohttp import web


class BaseTriggerServer:
    """
    Basic trigger server
    """

    def __init__(
        self,
        poller_fn: Callable,
        poller_args: Optional[list] = None,
        host: str = "0.0.0.0",
        port: int = 6789,
    ):
        """
        Args:
            poller_fn: function which will runs the polling of triggers. Will be run in a
                separate process. It's first arg must be a connection object it can
                use to communicate with the server process.
            poller_args: arguments that will be passed to poller_fn
            host: ip address to run server on
            port: port to run server on
        """

        self.host = host
        self.port = port
        self.poller_fn = poller_fn
        self.poller_args = poller_args or tuple()
        self.triggers = []
        self.process = None

    async def _check_triggers(self, request: web.Request) -> web.Response:
        """
        Function that is called to check if any triggers have been detected by pollers.
        """

        if self.triggers:
            out = web.Response(text=json.dumps(self.triggers.pop(0)))

            return out
        out = web.Response(text=json.dumps([]))

        return out

    async def _manual_trigger(self, request: web.Request) -> web.Response:
        """
        Allows users to post triggers directly to server for testing.
        """
        reqjson = await request.json()
        self.triggers.append(reqjson)

        return web.Response(text=json.dumps([]))

    async def poll_triggers(self):
        """
        Check whether the polling_fn has found anything
        """

        while True:
            if self.parent_conn.poll():
                self.triggers.append(self.parent_conn.recv())
            await asyncio.sleep(1)

    def get_routes(self) -> list:
        """
        Set up the server's routes.
        """

        return [
            web.get("/check_triggers", self._check_triggers),
            web.post("/trigger_manually", self._manual_trigger),
        ]

    async def main(self):
        """
        Run this in asyncio.run to run the server.
        """

        self.ctx = mp.get_context("spawn")
        self.run_process()
        app = web.Application()
        app.add_routes(self.get_routes())
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, host=self.host, port=self.port)
        await site.start()
        await self.poll_triggers()

    def __del__(self):
        """
        Clean up processes if server is deleted.
        """
        self.process.terminate()

    def run_process(self):
        """
        Run the polling process.
        """

        if self.process is not None:
            self.process.terminate()
            self.process.kill()
        parent_conn, child_conn = self.ctx.Pipe()
        self.parent_conn = parent_conn
        self.process = self.ctx.Process(
            target=self.poller_fn, args=(child_conn,) + self.poller_args
        )
        self.process.start()

    def run(self):
        asyncio.run(self.main())


def run_code(code_define: str, code_run: str) -> Any:
    """
    Run some code in a string and return the result

    Args:
        code_define (str): the code that does imports, function definitions, etc
        code_run (str): the one line whose result you want to output.
    """
    # adding the wrapper function is needed to get the imports to work
    # add indentation
    code_define = "\n    ".join(code_define.split("\n"))
    code_run = code_run.strip("\n")
    wrapper_fn = """
def wrapper():
    %s

    return %s
""" % (
        code_define,
        code_run,
    )
    exec(wrapper_fn)
    return eval("wrapper()")


def check_conditions(condition_registry: list[dict], code_registry: dict[str, dict]):
    """
    Checks all conditions once.
    """
    for condition in condition_registry:
        fn_name = condition["function_name"]
        code = code_registry[fn_name]
        status = run_code(code["code_define"], code["code_run"])
        if code["last_result"] != status:
            code["last_result"] = status
            if status == condition["notify_when"]:
                return condition
    return None


def condition_poller(
    conn: Connection, condition_registry: list[dict], code_registry: dict[str, dict]
):
    """
    Polls the registered conditions and runs the associated code.
    Intended to be run in subprocess. Communicates with the main process using conn.
    """
    while True:
        triggered_condition = check_conditions(condition_registry, code_registry)
        if triggered_condition:
            command = triggered_condition["action_description"]
            user = triggered_condition["user_name"]
            conn.send(
                {"command": command, "user": user, "conditions": condition_registry}
            )
        time.sleep(10)  # configure this for demo or whatever


class ConditionTriggerServer(BaseTriggerServer):
    """
    Trigger server where python code can be registered and run until a condition is met.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditions = []
        self.codes = {}
        self.poller_args = (self.conditions, self.codes)

    async def _add_condition(self, request: web.Request) -> web.Response:
        """
        Add a condition to be monitored by the server.
        """
        reqjson = await request.json()
        if "code" in reqjson:
            self.codes.update(reqjson["code"])
        self.conditions.append(reqjson["condition"])
        self.poller_args = (self.conditions, self.codes)
        # restart the poller process with the new conditions
        self.run_process()
        out = web.Response(text=json.dumps(["got it"]))
        return out

    async def _reset(self, request: web.Request) -> web.Response:
        """
        Wipe out all conditions.
        """
        self.conditions = []
        self.codes = {}
        self.poller_args = (self.conditions, self.codes)
        self.triggers = []
        return web.Response(text=json.dumps(["reset done"]))

    async def poll_triggers(self):
        """
        Check if the polling function has found anything.
        """
        while True:
            if self.parent_conn.poll():
                trigger = self.parent_conn.recv()
                trigger_out = {"user": trigger["user"], "command": trigger["command"]}
                self.triggers.append(trigger_out)
                # We only want to trigger on transitions, so we need to keep track of how
                # the condition status evolves. This means we need to synchronize the processes,
                # because the polling process will get restarted when new conditions are added.
                self.conditions = trigger["conditions"]
            await asyncio.sleep(1)

    def get_routes(self) -> list:
        return super().get_routes() + [
            web.post("/add_condition", self._add_condition),
            web.get("/reset", self._reset),
        ]


def run_server():
    """
    Run condition server in an asyncio loop.
    """
    url = os.environ["TRIGGER_SERVER_URL"]
    host, port = url.split(":")
    server = ConditionTriggerServer(condition_poller, host=host, port=port)
    asyncio.run(server.main())


class AllServerRunner:
    """
    Starts a new process and runs the trigger server.
    """

    def run(self):
        ctx = mp.get_context("spawn")
        self.process = ctx.Process(target=run_server)
        self.process.start()
        time.sleep(3)

    def __del__(self):
        self.process.terminate()
        self.process.kill()
