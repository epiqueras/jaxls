"""
Test for linting over LSP.
"""

from hamcrest import assert_that, is_

from .lsp_test_client import constants, session, utils

TEST_FILE_PATH = constants.TEST_DATA / "sample1" / "sample.py"
TEST_FILE_URI = utils.as_uri(str(TEST_FILE_PATH))
SERVER_INFO = utils.get_server_info_defaults()
TIMEOUT = 10  # 10 seconds


def test_example():
    """Test to linting on file open."""
    actual = {}
    with session.LspSession() as ls_session:
        ls_session.initialize()
        ls_session.set_notification_callback(session.WINDOW_LOG_MESSAGE, print)
        result = ls_session.text_document_inlay_hint(
            {
                "textDocument": {"uri": TEST_FILE_URI},
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 1, "character": 0},
                },
            }
        )
        # TODO: Implement test.
        print("RESULT", result)
    expected = {}
    assert_that(actual, is_(expected))
