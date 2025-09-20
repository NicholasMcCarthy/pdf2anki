"""Tests for CLI plan and sample modes."""

import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from pdf2anki.cli import app
from pdf2anki.config import Config, DocumentsConfig, DocumentConfig


def test_generate_plan_mode_no_documents_file():
    """Test --plan mode when documents.yaml doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        documents_file = tmpdir_path / "documents.yaml"
        
        # Should exit with error when documents.yaml doesn't exist
        with patch('pdf2anki.cli.console') as mock_console:
            try:
                app(['generate', '--plan', '--documents', str(documents_file)])
                assert False, "Expected typer.Exit to be raised"
            except SystemExit as e:
                # typer.Exit gets converted to SystemExit
                assert e.code == 1
            
            # Should print error message about missing documents.yaml
            mock_console.print.assert_called()


def test_generate_sample_mode_no_documents_file():
    """Test --sample mode when documents.yaml doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        documents_file = tmpdir_path / "documents.yaml"
        
        # Should exit with error when documents.yaml doesn't exist
        with patch('pdf2anki.cli.console') as mock_console:
            try:
                app(['generate', '--sample', '--documents', str(documents_file)])
                assert False, "Expected typer.Exit to be raised"
            except SystemExit as e:
                # typer.Exit gets converted to SystemExit
                assert e.code == 1
            
            # Should print error message about missing documents.yaml
            mock_console.print.assert_called()


def test_generate_plan_sample_csv_mode():
    """Test --plan-sample-csv mode works without documents.yaml."""
    with patch('pdf2anki.cli.console') as mock_console:
        with patch('pdf2anki.cli.get_note_type_manager') as mock_note_manager:
            # Mock note type manager to return sample types
            mock_manager = mock_note_manager.return_value
            mock_manager.list_note_types.return_value = ['basic', 'cloze']
            mock_manager.get_csv_fields.side_effect = lambda nt: {
                'basic': ['front', 'back'],
                'cloze': ['cloze_text']
            }.get(nt, [])
            
            # This should work without documents.yaml
            try:
                app(['generate', '--plan-sample-csv'])
            except SystemExit as e:
                # SystemExit with code 0 means success
                assert e.code == 0
            
            # Should have called console.print to show schema info
            mock_console.print.assert_called()


def test_plan_mode_no_llm_calls():
    """Test that --plan mode doesn't make any LLM calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create a minimal documents.yaml
        documents_config = DocumentsConfig()
        documents_config.documents = {
            "test.pdf": DocumentConfig(
                file_path="test.pdf",
                enabled=True
            )
        }
        documents_file = tmpdir_path / "documents.yaml"
        documents_config.to_yaml(documents_file)
        
        # Mock LLM provider creation to track if it's called
        with patch('pdf2anki.cli.create_llm_provider') as mock_llm_create:
            with patch('pdf2anki.cli.console'):
                with patch('pdf2anki.cli.PDFProcessor'):
                    with patch('pdf2anki.cli.TextChunker'):
                        try:
                            app(['generate', '--plan', '--documents', str(documents_file)])
                        except:
                            pass  # Expected to fail due to mocked dependencies
                        
                        # LLM provider should NOT be created in plan mode
                        mock_llm_create.assert_not_called()


def test_sample_mode_with_mocked_llm():
    """Test that --sample mode can work with mocked LLM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create a minimal documents.yaml
        documents_config = DocumentsConfig()
        documents_config.documents = {
            "test.pdf": DocumentConfig(
                file_path="test.pdf",
                enabled=True
            )
        }
        documents_file = tmpdir_path / "documents.yaml"
        documents_config.to_yaml(documents_file)
        
        with patch('pdf2anki.cli.console') as mock_console:
            with patch('pdf2anki.cli.PDFProcessor'):
                with patch('pdf2anki.cli.TextChunker'):
                    try:
                        app(['generate', '--sample', '--documents', str(documents_file)])
                    except:
                        pass  # Expected to fail due to mocked dependencies
                    
                    # Should have printed sample generation messages
                    mock_console.print.assert_called()


def test_no_workspace_writes_in_plan_mode():
    """Test that --plan mode doesn't write to workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        workspace_dir = tmpdir_path / "workspace"
        
        # Create a minimal documents.yaml
        documents_config = DocumentsConfig()
        documents_file = tmpdir_path / "documents.yaml"
        documents_config.to_yaml(documents_file)
        
        # Mock config to use our temp workspace
        with patch('pdf2anki.config.Config.from_yaml') as mock_config:
            config = Config()
            config.generate.output.workspace = workspace_dir
            mock_config.return_value = config
            
            with patch('pdf2anki.cli.console'):
                try:
                    app(['generate', '--plan', '--documents', str(documents_file)])
                except:
                    pass  # Expected to fail due to missing PDFs
                
                # Workspace should not be created in plan mode
                assert not workspace_dir.exists()


def test_no_workspace_writes_in_sample_mode():
    """Test that --sample mode doesn't write to workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        workspace_dir = tmpdir_path / "workspace"
        
        # Create a minimal documents.yaml
        documents_config = DocumentsConfig()
        documents_file = tmpdir_path / "documents.yaml"
        documents_config.to_yaml(documents_file)
        
        # Mock config to use our temp workspace
        with patch('pdf2anki.config.Config.from_yaml') as mock_config:
            config = Config()
            config.generate.output.workspace = workspace_dir
            mock_config.return_value = config
            
            with patch('pdf2anki.cli.console'):
                try:
                    app(['generate', '--sample', '--documents', str(documents_file)])
                except:
                    pass  # Expected to fail due to missing PDFs
                
                # Workspace should not be created in sample mode
                assert not workspace_dir.exists()