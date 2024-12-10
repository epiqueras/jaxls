"""Implementation of tool support over LSP."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import traceback
from typing import Any, Optional


# **********************************************************
# Update sys.path before importing any bundled libraries.
# **********************************************************
def update_sys_path(path_to_add: str, strategy: str) -> None:
    """Add given path to `sys.path`."""
    if path_to_add not in sys.path and os.path.isdir(path_to_add):
        if strategy == "useBundled":
            sys.path.insert(0, path_to_add)
        elif strategy == "fromEnvironment":
            sys.path.append(path_to_add)


# Ensure that we can import LSP libraries, and other bundled libraries.
update_sys_path(
    os.fspath(pathlib.Path(__file__).parent.parent / "libs"),
    os.getenv("LS_IMPORT_STRATEGY", "fromEnvironment"),
)

# **********************************************************
# Imports needed for the language server goes below this.
# **********************************************************
import jaxls  # noqa: E402
import lsp_jsonrpc as jsonrpc  # noqa: E402
import lsp_utils as utils  # noqa: E402
import lsprotocol.types as lsp  # noqa: E402
from pygls import server, uris, workspace  # noqa: E402

WORKSPACE_SETTINGS = {}
GLOBAL_SETTINGS = {}

MAX_WORKERS = 5
LSP_SERVER = server.LanguageServer(
    name="JAX Language Server", version="2024.0.0-dev", max_workers=MAX_WORKERS
)

TOOL_MODULE = "jaxls"
TOOL_DISPLAY = "JAX Language Server"
TOOL_ARGS = []


# **********************************************************
# Required Language Server Initialization and Exit handlers.
# **********************************************************
@LSP_SERVER.feature(lsp.INITIALIZE)
def initialize(params: lsp.InitializeParams) -> None:
    """LSP handler for initialize request."""
    log_to_output(f"CWD Server: {os.getcwd()}")

    paths = "\r\n   ".join(sys.path)
    log_to_output(f"sys.path used to run Server:\r\n   {paths}")

    assert params.initialization_options is not None
    GLOBAL_SETTINGS.update(**params.initialization_options.get("globalSettings", {}))

    settings = params.initialization_options["settings"]
    _update_workspace_settings(settings)
    log_to_output(
        f"Settings used to run Server:\r\n{json.dumps(settings, indent=4, ensure_ascii=False)}\r\n"
    )
    log_to_output(
        f"Global settings:\r\n{json.dumps(GLOBAL_SETTINGS, indent=4, ensure_ascii=False)}\r\n"
    )


@LSP_SERVER.feature(lsp.EXIT)
def on_exit(_params: Optional[Any] = None) -> None:
    """Handle clean up on exit."""
    jsonrpc.shutdown_json_rpc()


@LSP_SERVER.feature(lsp.SHUTDOWN)
def on_shutdown(_params: Optional[Any] = None) -> None:
    """Handle clean up on shutdown."""
    jsonrpc.shutdown_json_rpc()


def _get_global_defaults():
    return {
        "path": GLOBAL_SETTINGS.get("path", []),
        "interpreter": GLOBAL_SETTINGS.get("interpreter", [sys.executable]),
        "args": GLOBAL_SETTINGS.get("args", []),
        "importStrategy": GLOBAL_SETTINGS.get("importStrategy", "fromEnvironment"),
        "showNotifications": GLOBAL_SETTINGS.get("showNotifications", "off"),
    }


def _update_workspace_settings(settings):
    if not settings:
        key = os.getcwd()
        WORKSPACE_SETTINGS[key] = {
            "cwd": key,
            "workspaceFS": key,
            "workspace": uris.from_fs_path(key),
            **_get_global_defaults(),
        }
        return

    for setting in settings:
        key = uris.to_fs_path(setting["workspace"])
        WORKSPACE_SETTINGS[key] = {
            "cwd": key,
            **setting,
            "workspaceFS": key,
        }


def _get_settings_by_path(file_path: pathlib.Path):
    workspaces = {s["workspaceFS"] for s in WORKSPACE_SETTINGS.values()}

    while file_path != file_path.parent:
        str_file_path = str(file_path)
        if str_file_path in workspaces:
            return WORKSPACE_SETTINGS[str_file_path]
        file_path = file_path.parent

    setting_values = list(WORKSPACE_SETTINGS.values())
    return setting_values[0]


def _get_document_key(document: workspace.Document):
    if WORKSPACE_SETTINGS:
        document_workspace = pathlib.Path(document.path)
        workspaces = {s["workspaceFS"] for s in WORKSPACE_SETTINGS.values()}

        # Find workspace settings for the given file.
        while document_workspace != document_workspace.parent:
            if str(document_workspace) in workspaces:
                return str(document_workspace)
            document_workspace = document_workspace.parent

    return None


def _get_settings_by_document(document: workspace.Document | None):
    if document is None or document.path is None:
        return list(WORKSPACE_SETTINGS.values())[0]

    key = _get_document_key(document)
    if key is None:
        # This is either a non-workspace file or there is no workspace.
        key = os.fspath(pathlib.Path(document.path).parent)
        return {
            "cwd": key,
            "workspaceFS": key,
            "workspace": uris.from_fs_path(key),
            **_get_global_defaults(),
        }

    return WORKSPACE_SETTINGS[str(key)]


# *****************************************************
# Logging and notification.
# *****************************************************
def log_to_output(
    message: str, msg_type: lsp.MessageType = lsp.MessageType.Log
) -> None:
    LSP_SERVER.show_message_log(message, msg_type)


def log_error(message: str) -> None:
    LSP_SERVER.show_message_log(message, lsp.MessageType.Error)
    if os.getenv("LS_SHOW_NOTIFICATION", "off") in ["onError", "onWarning", "always"]:
        LSP_SERVER.show_message(message, lsp.MessageType.Error)


def log_warning(message: str) -> None:
    LSP_SERVER.show_message_log(message, lsp.MessageType.Warning)
    if os.getenv("LS_SHOW_NOTIFICATION", "off") in ["onWarning", "always"]:
        LSP_SERVER.show_message(message, lsp.MessageType.Warning)


def log_always(message: str) -> None:
    LSP_SERVER.show_message_log(message, lsp.MessageType.Info)
    if os.getenv("LS_SHOW_NOTIFICATION", "off") in ["always"]:
        LSP_SERVER.show_message(message, lsp.MessageType.Info)


# *****************************************************
# Internal execution APIs.
# *****************************************************
def _run_tool(
    document: workspace.Document,
    extra_args: Optional[list[str]] = None,
    use_stdin: bool = False,
) -> utils.RunResult | jsonrpc.RpcRunResult | None:
    """Runs tool on the given document.

    if use_stdin is true then contents of the document is passed to the
    tool via stdin.
    """
    if extra_args is None:
        extra_args = []
    if str(document.uri).startswith("vscode-notebook-cell"):
        # Skip notebook cells
        return None
    if utils.is_stdlib_file(document.path):
        # Skip standard library python files.
        return None

    # Deep copy here to prevent accidentally updating global settings.
    settings = copy.deepcopy(_get_settings_by_document(document))

    code_workspace = settings["workspaceFS"]
    cwd = settings["cwd"]

    use_path = False
    use_rpc = False
    if settings["path"]:
        # 'path' setting takes priority over everything.
        use_path = True
        argv = settings["path"]
    elif settings["interpreter"] and not utils.is_current_interpreter(
        settings["interpreter"][0]
    ):
        # If there is a different interpreter set use JSON-RPC to the subprocess
        # running under that interpreter.
        argv = [TOOL_MODULE]
        use_rpc = True
    else:
        # if the interpreter is same as the interpreter running this
        # process then run as module.
        argv = [TOOL_MODULE]

    argv += extra_args + [document.path] + TOOL_ARGS + settings["args"]
    if use_stdin:
        argv += ["--use-stdin"]
    document_source = document.source.replace("\r\n", "\n") if use_stdin else None

    if use_path:
        # This mode is used when running executables.
        log_to_output(" ".join(argv))
        log_to_output(f"CWD Server: {cwd}")
        result = utils.run_path(
            argv=argv, use_stdin=use_stdin, cwd=cwd, source=document_source
        )
        if result.stderr:
            log_to_output(result.stderr)
    elif use_rpc:
        # This mode is used if the interpreter running this server is different from
        # the interpreter used for running this server.
        log_to_output(" ".join(settings["interpreter"] + ["-m"] + argv))
        log_to_output(f"CWD Server: {cwd}")
        result = jsonrpc.run_over_json_rpc(
            workspace=code_workspace,
            interpreter=settings["interpreter"],
            module=TOOL_MODULE,
            argv=argv,
            use_stdin=use_stdin,
            cwd=cwd,
            source=document_source,
        )
        if result.exception:
            log_error(result.exception)
            result = utils.RunResult(result.stdout, result.stderr)
        elif result.stderr:
            log_to_output(result.stderr)
    else:
        # In this mode the tool is run as a module in the same process as the language server.
        log_to_output(" ".join([sys.executable, "-m"] + argv))
        log_to_output(f"CWD Server: {cwd}")
        # This is needed to preserve sys.path, in cases where the tool modifies
        # sys.path and that might not work for this scenario next time around.
        with utils.substitute_attr(sys, "path", sys.path[:]):
            try:
                result = utils.run_module(
                    module=TOOL_MODULE,
                    argv=argv,
                    use_stdin=use_stdin,
                    cwd=cwd,
                    source=document_source,
                )
            except Exception:
                log_error(traceback.format_exc(chain=True))
                raise
        if result.stderr:
            log_to_output(result.stderr)

    log_to_output(f"{document.uri} :\r\n{result.stdout}")
    return result


# *****************************************************
# Typed LSP features.
# *****************************************************


class TypedRunner:
    def __init__(self):
        self.type_registry = jaxls.TypeRegistry(frames={})

    def run_on_document(
        self, document: workspace.TextDocument, use_stdin: bool = False
    ) -> jaxls.TypeRegistry:
        result = _run_tool(document, [jaxls.Method.typed], use_stdin=use_stdin)
        if result is None:
            return jaxls.TypeRegistry(frames={})
        return jaxls.type_registry.model_validate_json(result.stdout)

    def on_open(self, document: workspace.TextDocument):
        new_type_registry = self.run_on_document(document)
        self.type_registry.frames.update(new_type_registry.frames)

    def on_change(
        self, document: workspace.TextDocument, params: lsp.DidChangeTextDocumentParams
    ):
        new_type_registry = self.run_on_document(document, use_stdin=True)
        keep_prefix_frames = False
        if document.path in new_type_registry.frames:
            for eqn in new_type_registry.frames[document.path].values():
                if eqn.singleton:
                    keep_prefix_frames = True
                    break

        # 0 indexed line number.
        first_changed_line = sys.maxsize
        for change in params.content_changes:
            if isinstance(change, lsp.TextDocumentContentChangeEvent_Type1):
                first_changed_line = min(first_changed_line, change.range.start.line)
        if first_changed_line == sys.maxsize or not keep_prefix_frames:
            self.type_registry.frames.update(new_type_registry.frames)
            return

        # Remove frames on or after the line that changed.
        keys_to_delete = []
        for key, eqn in self.type_registry.frames[document.path].items():
            if eqn.frame.start_line - 1 >= first_changed_line:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.type_registry.frames[document.path][key]

        # Merge new frames with old frames of current document.
        new_type_registry.frames[document.path].update(
            self.type_registry.frames[document.path]
        )
        self.type_registry.frames.update(new_type_registry.frames)

    def get_inlay_hints(
        self, document: workspace.TextDocument
    ) -> list[lsp.InlayHint] | None:
        if document.path not in self.type_registry.frames:
            self.on_open(document)
        if document.path not in self.type_registry.frames:
            return None
        inlay_hints: list[lsp.InlayHint] = []
        for eqn_types in self.type_registry.frames[document.path].values():
            frame = eqn_types.frame
            if eqn_types.message:
                label = f": {eqn_types.message}"
                tooltip = eqn_types.tooltip or eqn_types.message
            else:
                shape = eqn_types.out_shapes[-1]
                label = f": {jaxls.pp_shapes(shape)}"
                tooltip = eqn_types.tooltip or label
            inlay_hints.append(
                lsp.InlayHint(
                    position=lsp.Position(
                        line=frame.start_line - 1, character=frame.end_column
                    ),
                    label=label,
                    kind=lsp.InlayHintKind.Type,
                    tooltip=tooltip,
                    padding_left=False,
                    padding_right=True,
                )
            )
        return inlay_hints


TYPED_RUNNER = TypedRunner()


@LSP_SERVER.feature(lsp.TEXT_DOCUMENT_INLAY_HINT)
def inlay_hints(params: lsp.InlayHintParams):
    document = LSP_SERVER.workspace.get_text_document(params.text_document.uri)
    log_to_output("Called inlay hints.")
    return TYPED_RUNNER.get_inlay_hints(document)


# *****************************************************
# Document handlers.
# *****************************************************


@LSP_SERVER.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(params: lsp.DidOpenTextDocumentParams):
    document = LSP_SERVER.workspace.get_text_document(params.text_document.uri)
    TYPED_RUNNER.on_open(document)


@LSP_SERVER.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(params: lsp.DidChangeTextDocumentParams):
    document = LSP_SERVER.workspace.get_text_document(params.text_document.uri)
    TYPED_RUNNER.on_change(document, params)


# *****************************************************
# Start the server.
# *****************************************************
if __name__ == "__main__":
    LSP_SERVER.start_io()
