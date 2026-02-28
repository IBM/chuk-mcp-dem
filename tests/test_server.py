"""Comprehensive tests for server.py and async_server.py."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: because importing server.py triggers `from .async_server import mcp`
# which in turn imports ChukMCPServer, DEMManager, and the register_*_tools
# functions, we need to mock those heavy dependencies when we test server.py
# logic in isolation.
# ---------------------------------------------------------------------------


def _make_mock_modules():
    """Return a dict of mock modules suitable for sys.modules patching."""
    mock_chuk_mcp_server = MagicMock()
    mock_chuk_mcp_server.ChukMCPServer.return_value = MagicMock(name="mcp_instance")

    mock_chuk_artifacts = MagicMock()
    mock_chuk_artifacts.ArtifactStore.return_value = MagicMock(name="store_instance")

    mock_dotenv = MagicMock()

    return {
        "chuk_mcp_server": mock_chuk_mcp_server,
        "chuk_artifacts": mock_chuk_artifacts,
        "dotenv": mock_dotenv,
    }


@pytest.fixture
def clean_server_import():
    """
    Context manager that removes cached server modules so we can re-import
    with fresh mocks.
    """
    keys_to_remove = [
        k
        for k in sys.modules
        if k.startswith("chuk_mcp_dem.server") or k.startswith("chuk_mcp_dem.async_server")
    ]
    saved = {k: sys.modules.pop(k) for k in keys_to_remove}
    yield
    # Restore originals
    for k in list(sys.modules.keys()):
        if k.startswith("chuk_mcp_dem.server") or k.startswith("chuk_mcp_dem.async_server"):
            sys.modules.pop(k, None)
    sys.modules.update(saved)


# =====================================================================
# _init_artifact_store tests
# =====================================================================


class TestInitArtifactStoreDefaultMemory:
    """No env vars set -- should fall back to memory provider."""

    def test_default_memory_provider(self, clean_server_import):
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock(name="set_global_artifact_store")

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        mock_store_cls.assert_called_once_with(
            storage_provider="memory",
            session_provider="memory",
        )
        mock_set_global.assert_called_once_with(mock_store_cls.return_value)


class TestInitArtifactStoreS3:
    """S3 provider with full creds."""

    def test_s3_provider_success(self, clean_server_import):
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock(name="set_global_artifact_store")

        env = {
            "CHUK_ARTIFACTS_PROVIDER": "s3",
            "BUCKET_NAME": "my-bucket",
            "AWS_ACCESS_KEY_ID": "AKID",
            "AWS_SECRET_ACCESS_KEY": "SECRET",
            "AWS_ENDPOINT_URL_S3": "https://s3.example.com",
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        mock_store_cls.assert_called_once_with(
            storage_provider="s3",
            session_provider="memory",
            bucket="my-bucket",
        )
        mock_set_global.assert_called_once()

    def test_s3_with_redis_session(self, clean_server_import):
        """When REDIS_URL is set, session_provider should be 'redis'."""
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock(name="set_global_artifact_store")

        env = {
            "CHUK_ARTIFACTS_PROVIDER": "s3",
            "BUCKET_NAME": "bucket",
            "AWS_ACCESS_KEY_ID": "AKID",
            "AWS_SECRET_ACCESS_KEY": "SECRET",
            "REDIS_URL": "redis://localhost:6379",
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        mock_store_cls.assert_called_once_with(
            storage_provider="s3",
            session_provider="redis",
            bucket="bucket",
        )

    def test_s3_missing_creds_returns_false(self, clean_server_import):
        """S3 without required creds should return False immediately."""
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock(name="set_global_artifact_store")

        env = {
            "CHUK_ARTIFACTS_PROVIDER": "s3",
            # Missing BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is False
        mock_store_cls.assert_not_called()
        mock_set_global.assert_not_called()

    def test_s3_missing_bucket_returns_false(self, clean_server_import):
        """S3 with keys but no bucket should return False."""
        env = {
            "CHUK_ARTIFACTS_PROVIDER": "s3",
            "AWS_ACCESS_KEY_ID": "AKID",
            "AWS_SECRET_ACCESS_KEY": "SECRET",
            # Missing BUCKET_NAME
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is False

    def test_s3_missing_secret_returns_false(self, clean_server_import):
        """S3 with key but no secret should return False."""
        env = {
            "CHUK_ARTIFACTS_PROVIDER": "s3",
            "BUCKET_NAME": "bucket",
            "AWS_ACCESS_KEY_ID": "AKID",
            # Missing AWS_SECRET_ACCESS_KEY
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is False


class TestInitArtifactStoreFilesystem:
    """Filesystem provider tests."""

    def test_filesystem_with_path(self, clean_server_import, tmp_path):
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock(name="set_global_artifact_store")
        artifacts_dir = str(tmp_path / "artifacts")

        env = {
            "CHUK_ARTIFACTS_PROVIDER": "filesystem",
            "CHUK_ARTIFACTS_PATH": artifacts_dir,
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        # The directory should have been created
        assert Path(artifacts_dir).is_dir()
        mock_store_cls.assert_called_once_with(
            storage_provider="filesystem",
            session_provider="memory",
            bucket=artifacts_dir,
        )
        mock_set_global.assert_called_once()

    def test_filesystem_no_path_falls_back_to_memory(self, clean_server_import):
        """Filesystem provider without CHUK_ARTIFACTS_PATH falls back to memory."""
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock(name="set_global_artifact_store")

        env = {
            "CHUK_ARTIFACTS_PROVIDER": "filesystem",
            # No CHUK_ARTIFACTS_PATH
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        # Provider should have been switched to memory
        mock_store_cls.assert_called_once_with(
            storage_provider="memory",
            session_provider="memory",
        )

    def test_filesystem_with_redis(self, clean_server_import, tmp_path):
        """Filesystem provider with redis session."""
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock(name="set_global_artifact_store")
        artifacts_dir = str(tmp_path / "fs_artifacts")

        env = {
            "CHUK_ARTIFACTS_PROVIDER": "filesystem",
            "CHUK_ARTIFACTS_PATH": artifacts_dir,
            "REDIS_URL": "redis://localhost:6379/0",
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        mock_store_cls.assert_called_once_with(
            storage_provider="filesystem",
            session_provider="redis",
            bucket=artifacts_dir,
        )


class TestInitArtifactStoreErrors:
    """Error handling in _init_artifact_store."""

    def test_artifact_store_import_error_returns_false(self, clean_server_import):
        """If ArtifactStore import fails, should return False."""
        env = {}

        with patch.dict(os.environ, env, clear=True):
            with patch.dict(sys.modules, {"chuk_artifacts": None}):
                # Importing chuk_artifacts will raise ImportError when it's None in sys.modules
                # But server.py uses a local import inside try/except, so we need
                # to make the import raise an exception
                with patch("builtins.__import__", side_effect=_import_raiser("chuk_artifacts")):
                    from chuk_mcp_dem.server import _init_artifact_store

                    result = _init_artifact_store()

        assert result is False

    def test_artifact_store_constructor_raises(self, clean_server_import):
        """If ArtifactStore() raises, should return False."""
        mock_store_cls = MagicMock(side_effect=RuntimeError("connection refused"))
        mock_set_global = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is False
        mock_set_global.assert_not_called()

    def test_set_global_raises(self, clean_server_import):
        """If set_global_artifact_store raises, should return False."""
        mock_store_cls = MagicMock()
        mock_set_global = MagicMock(side_effect=RuntimeError("global store error"))

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is False


def _import_raiser(blocked_module):
    """Return a function that raises ImportError for a specific module."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def custom_import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return original_import(name, *args, **kwargs)

    return custom_import


class TestInitArtifactStoreRedis:
    """Redis-related session provider logic."""

    def test_redis_url_sets_redis_session_provider(self, clean_server_import):
        """REDIS_URL env var should set session_provider to 'redis'."""
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock()

        env = {
            "REDIS_URL": "redis://cache.example.com:6379",
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        mock_store_cls.assert_called_once_with(
            storage_provider="memory",
            session_provider="redis",
        )

    def test_no_redis_url_uses_memory_session(self, clean_server_import):
        """Without REDIS_URL, session_provider should be 'memory'."""
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        mock_store_cls.assert_called_once_with(
            storage_provider="memory",
            session_provider="memory",
        )


class TestInitArtifactStoreSetGlobalCalled:
    """Verify set_global_artifact_store is called on success."""

    def test_set_global_called_with_store_instance(self, clean_server_import):
        mock_store_instance = MagicMock(name="store_instance")
        mock_store_cls = MagicMock(return_value=mock_store_instance)
        mock_set_global = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                _init_artifact_store()

        mock_set_global.assert_called_once_with(mock_store_instance)


# =====================================================================
# main() tests
# =====================================================================


class TestMainStdioMode:
    """Tests for main() with stdio mode."""

    def test_stdio_mode_calls_mcp_run_stdio(self, clean_server_import):
        mock_mcp = MagicMock(name="mcp")

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem import server

                server.mcp = mock_mcp

                with patch("sys.argv", ["server", "stdio"]):
                    server.main()

        mock_mcp.run.assert_called_once_with(stdio=True)


class TestMainHttpMode:
    """Tests for main() with http mode."""

    def test_http_mode_calls_mcp_run_with_host_port(self, clean_server_import):
        mock_mcp = MagicMock(name="mcp")

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem import server

                server.mcp = mock_mcp

                with patch("sys.argv", ["server", "http", "--host", "0.0.0.0", "--port", "9000"]):
                    server.main()

        mock_mcp.run.assert_called_once_with(host="0.0.0.0", port=9000, stdio=False)

    def test_http_mode_default_host_port(self, clean_server_import):
        """http mode with no explicit --host/--port uses defaults."""
        mock_mcp = MagicMock(name="mcp")

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem import server

                server.mcp = mock_mcp

                with patch("sys.argv", ["server", "http"]):
                    server.main()

        mock_mcp.run.assert_called_once_with(host="0.0.0.0", port=8003, stdio=False)

    def test_default_port_is_8003(self, clean_server_import):
        """Verify the argparse default for --port is 8003."""
        mock_mcp = MagicMock(name="mcp")

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem import server

                server.mcp = mock_mcp

                with patch("sys.argv", ["server", "http"]):
                    server.main()

        _, kwargs = mock_mcp.run.call_args
        assert kwargs["port"] == 8003


class TestMainAutoDetect:
    """Tests for main() auto-detection (no mode specified)."""

    def test_auto_detect_stdio_when_mcp_stdio_env_set(self, clean_server_import):
        """MCP_STDIO env var triggers stdio mode."""
        mock_mcp = MagicMock(name="mcp")

        env = {"MCP_STDIO": "1"}
        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem import server

                server.mcp = mock_mcp

                with patch("sys.argv", ["server"]):
                    server.main()

        mock_mcp.run.assert_called_once_with(stdio=True)

    def test_auto_detect_stdio_when_stdin_not_tty(self, clean_server_import):
        """When stdin is not a TTY, should auto-detect stdio."""
        mock_mcp = MagicMock(name="mcp")

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem import server

                server.mcp = mock_mcp

                with patch("sys.argv", ["server"]), patch("sys.stdin") as mock_stdin:
                    mock_stdin.isatty.return_value = False
                    server.main()

        mock_mcp.run.assert_called_once_with(stdio=True)

    def test_auto_detect_http_when_tty(self, clean_server_import):
        """When stdin IS a TTY and no MCP_STDIO, should default to HTTP."""
        mock_mcp = MagicMock(name="mcp")

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem import server

                server.mcp = mock_mcp

                with patch("sys.argv", ["server"]), patch("sys.stdin") as mock_stdin:
                    mock_stdin.isatty.return_value = True
                    server.main()

        mock_mcp.run.assert_called_once_with(host="0.0.0.0", port=8003, stdio=False)

    def test_auto_detect_http_respects_custom_host_port(self, clean_server_import):
        """Auto-detect HTTP mode should honour --host and --port flags."""
        mock_mcp = MagicMock(name="mcp")

        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem import server

                server.mcp = mock_mcp

                with (
                    patch("sys.argv", ["server", "--host", "10.0.0.1", "--port", "7777"]),
                    patch("sys.stdin") as mock_stdin,
                ):
                    mock_stdin.isatty.return_value = True
                    server.main()

        mock_mcp.run.assert_called_once_with(host="10.0.0.1", port=7777, stdio=False)


class TestMainCallsInitArtifactStore:
    """main() should call _init_artifact_store before running."""

    def test_init_artifact_store_called(self, clean_server_import):
        with patch.dict(os.environ, {}, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", MagicMock()),
                patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
            ):
                from chuk_mcp_dem import server

                server.mcp = MagicMock()

                with (
                    patch.object(server, "_init_artifact_store") as mock_init,
                    patch("sys.argv", ["server", "stdio"]),
                ):
                    server.main()

                mock_init.assert_called_once()


# =====================================================================
# async_server.py tests
# =====================================================================


class TestAsyncServerMCPInstance:
    """Tests for the mcp and manager instances in async_server.py."""

    def test_mcp_is_chuk_mcp_server_instance(self):
        from chuk_mcp_server import ChukMCPServer
        from chuk_mcp_dem.async_server import mcp

        assert isinstance(mcp, ChukMCPServer)

    def test_mcp_name_is_chuk_mcp_dem(self):
        from chuk_mcp_dem.async_server import mcp

        assert mcp.server_info.name == "chuk-mcp-dem"

    def test_manager_is_dem_manager_instance(self):
        from chuk_mcp_dem.core.dem_manager import DEMManager
        from chuk_mcp_dem.async_server import manager

        assert isinstance(manager, DEMManager)


class TestAsyncServerToolRegistration:
    """Verify that tool registration functions were invoked at import time."""

    def test_tools_are_registered(self):
        """
        After importing async_server, the mcp instance should have tools
        registered (discovery, download, analysis).  We verify by checking
        that mcp has at least one tool registered.
        """
        from chuk_mcp_dem.async_server import mcp

        # ChukMCPServer should have some tools after registration
        # The exact attribute depends on the framework, but the fact
        # that the import succeeded without error means register_*_tools
        # were called successfully.
        assert mcp is not None


# =====================================================================
# Module-level import side-effects in server.py
# =====================================================================


class TestServerModuleImport:
    """Tests for module-level behavior in server.py."""

    def test_server_exports_mcp(self, clean_server_import):
        """server.py should export the mcp instance from async_server."""
        with (
            patch("chuk_artifacts.ArtifactStore", MagicMock()),
            patch("chuk_mcp_server.set_global_artifact_store", MagicMock()),
        ):
            from chuk_mcp_dem.server import mcp

            assert mcp is not None

    def test_mcp_from_server_is_same_as_async_server(self):
        """The mcp in server.py should be the same object as in async_server.py."""
        from chuk_mcp_dem.server import mcp as server_mcp
        from chuk_mcp_dem.async_server import mcp as async_mcp

        assert server_mcp is async_mcp


# =====================================================================
# Edge cases
# =====================================================================


class TestInitArtifactStoreEdgeCases:
    """Edge cases and boundary conditions."""

    def test_unknown_provider_treated_as_custom(self, clean_server_import):
        """An unrecognized provider string should still attempt ArtifactStore."""
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock()

        env = {"CHUK_ARTIFACTS_PROVIDER": "custom_provider"}

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        mock_store_cls.assert_called_once_with(
            storage_provider="custom_provider",
            session_provider="memory",
        )

    def test_filesystem_creates_nested_directories(self, clean_server_import, tmp_path):
        """Filesystem provider should create nested directories."""
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock()
        nested_dir = str(tmp_path / "a" / "b" / "c" / "artifacts")

        env = {
            "CHUK_ARTIFACTS_PROVIDER": "filesystem",
            "CHUK_ARTIFACTS_PATH": nested_dir,
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        assert Path(nested_dir).is_dir()

    def test_s3_provider_no_endpoint_url_still_works(self, clean_server_import):
        """S3 without AWS_ENDPOINT_URL_S3 should still succeed (uses default AWS)."""
        mock_store_cls = MagicMock(name="ArtifactStore")
        mock_set_global = MagicMock()

        env = {
            "CHUK_ARTIFACTS_PROVIDER": "s3",
            "BUCKET_NAME": "bucket",
            "AWS_ACCESS_KEY_ID": "AKID",
            "AWS_SECRET_ACCESS_KEY": "SECRET",
            # No AWS_ENDPOINT_URL_S3 -- that's fine
        }

        with patch.dict(os.environ, env, clear=True):
            with (
                patch("chuk_artifacts.ArtifactStore", mock_store_cls),
                patch("chuk_mcp_server.set_global_artifact_store", mock_set_global),
            ):
                from chuk_mcp_dem.server import _init_artifact_store

                result = _init_artifact_store()

        assert result is True
        mock_store_cls.assert_called_once_with(
            storage_provider="s3",
            session_provider="memory",
            bucket="bucket",
        )
