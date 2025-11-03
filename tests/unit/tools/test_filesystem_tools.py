# tests/unit/test_filesystem_tools.py

from pathlib import Path

import pytest

from allos.tools.filesystem.directory import ListDirectoryTool
from allos.tools.filesystem.edit import FileEditTool

# Import the tools to ensure they are registered for testing
from allos.tools.filesystem.read import FileReadTool
from allos.tools.filesystem.write import FileWriteTool


class TestFileReadTool:
    def setup_method(self):
        self.tool = FileReadTool()
        self.sample_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

    def test_read_entire_file(self, work_dir: Path):
        """Test reading the full content of a file."""
        (work_dir / "test.txt").write_text(self.sample_content)
        result = self.tool.execute(path="test.txt")
        assert result["status"] == "success"
        assert result["content"] == self.sample_content

    def test_read_line_range(self, work_dir: Path):
        """Test reading a specific range of lines (2-4)."""
        (work_dir / "test.txt").write_text(self.sample_content)
        result = self.tool.execute(path="test.txt", start_line=2, end_line=4)
        assert result["status"] == "success"
        assert result["content"] == "Line 2\nLine 3"

    def test_read_from_start_line(self, work_dir: Path):
        """Test reading from a start line to the end."""
        (work_dir / "test.txt").write_text(self.sample_content)
        result = self.tool.execute(path="test.txt", start_line=4)
        assert result["status"] == "success"
        assert result["content"] == "Line 4\nLine 5"

    def test_read_file_not_found(self, work_dir: Path):
        """Test reading a file that does not exist."""
        result = self.tool.execute(path="non_existent.txt")
        assert result["status"] == "error"
        assert "File not found" in result["message"]

    def test_read_unsafe_path_fails(self, work_dir: Path):
        """Test that attempting to read a file outside the work_dir fails."""
        # Create a file outside the safe directory
        (work_dir.parent / "secret.txt").write_text("sensitive data")

        result = self.tool.execute(path="../secret.txt")
        assert result["status"] == "error"
        assert "is outside the safe working directory" in result["message"]

    def test_execute_missing_path_returns_error(self):
        """Test that execute returns an error if 'path' is not provided."""
        result = self.tool.execute(start_line=1)
        assert result["status"] == "error"
        assert "'path' argument is required" in result["message"]

    def test_read_invalid_line_range(self, work_dir: Path):
        """Test reading a range where start_line is not less than end_line."""
        (work_dir / "test.txt").write_text(self.sample_content)
        # Test with start_line == end_line
        result = self.tool.execute(path="test.txt", start_line=3, end_line=3)
        assert result["status"] == "success"
        assert result["content"] == ""
        assert "Invalid line range" in result["message"]

        # Test with start_line > end_line
        result2 = self.tool.execute(path="test.txt", start_line=4, end_line=2)
        assert result2["status"] == "success"
        assert result2["content"] == ""
        assert "Invalid line range" in result2["message"]


class TestFileWriteTool:
    def setup_method(self):
        self.tool = FileWriteTool()

    def test_write_new_file(self, work_dir: Path):
        """Test writing content to a new file."""
        file_path = "new_file.txt"
        content = "Hello, Allos!"

        result = self.tool.execute(path=file_path, content=content)
        assert result["status"] == "success"

        # Verify the file was created with the correct content
        created_file = work_dir / file_path
        assert created_file.exists()
        assert created_file.read_text() == content

    def test_overwrite_existing_file(self, work_dir: Path):
        """Test that writing to an existing file overwrites its content."""
        file_path = "existing.txt"
        (work_dir / file_path).write_text("Original content")

        new_content = "Overwritten content"
        result = self.tool.execute(path=file_path, content=new_content)
        assert result["status"] == "success"

        assert (work_dir / file_path).read_text() == new_content

    def test_write_to_new_subdirectory(self, work_dir: Path):
        """Test writing a file that requires creating a new directory."""
        file_path = "new_dir/another_file.txt"
        content = "Content in a subdir"

        result = self.tool.execute(path=file_path, content=content)
        assert result["status"] == "success"

        created_file = work_dir / file_path
        assert created_file.exists()
        assert created_file.read_text() == content

    def test_write_unsafe_path_fails(self, work_dir: Path):
        """Test that attempting to write a file outside the work_dir fails."""
        result = self.tool.execute(path="../unsafe_file.txt", content="danger")
        assert result["status"] == "error"
        assert "is outside the safe working directory" in result["message"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"content": "some content"}, id="missing_path"),
            pytest.param({"path": "some_file.txt"}, id="missing_content"),
            pytest.param({}, id="missing_both"),
        ],
    )
    def test_execute_missing_args_returns_error(self, kwargs):
        """Test that execute returns an error if 'path' or 'content' is missing."""
        result = self.tool.execute(**kwargs)
        assert result["status"] == "error"
        assert "Both 'path' and 'content' arguments are required" in result["message"]


class TestFileEditTool:
    def setup_method(self):
        self.tool = FileEditTool()
        self.original_content = "Hello, world! This is a test."

    def test_edit_successful(self, work_dir: Path):
        """Test a successful, unique find-and-replace operation."""
        file_path = "edit_me.txt"
        (work_dir / file_path).write_text(self.original_content)

        result = self.tool.execute(
            path=file_path, find_string="world", replace_with="Allos"
        )
        assert result["status"] == "success"

        # Verify file content was changed
        edited_content = (work_dir / file_path).read_text()
        assert edited_content == "Hello, Allos! This is a test."

    def test_edit_find_string_not_found(self, work_dir: Path):
        """Test that the edit fails if the find_string is not found."""
        file_path = "edit_me.txt"
        (work_dir / file_path).write_text(self.original_content)

        result = self.tool.execute(
            path=file_path, find_string="galaxy", replace_with="cosmos"
        )
        assert result["status"] == "error"
        assert "was not found" in result["message"]

    def test_edit_find_string_not_unique(self, work_dir: Path):
        """Test that the edit fails if the find_string appears more than once."""
        file_path = "edit_me.txt"
        content_with_duplicates = "test test test"
        (work_dir / file_path).write_text(content_with_duplicates)

        result = self.tool.execute(
            path=file_path, find_string="test", replace_with="exam"
        )
        assert result["status"] == "error"
        assert "appeared 3 times" in result["message"]
        assert "not unique" in result["message"]

    def test_edit_unsafe_path_fails(self, work_dir: Path):
        """Test that editing an unsafe path fails."""
        result = self.tool.execute(
            path="../unsafe.txt", find_string="a", replace_with="b"
        )
        assert result["status"] == "error"
        assert "is outside the safe working directory" in result["message"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"find_string": "a", "replace_with": "b"}, id="missing_path"),
            pytest.param(
                {"path": "f.txt", "replace_with": "b"}, id="missing_find_string"
            ),
            pytest.param(
                {"path": "f.txt", "find_string": "a"}, id="missing_replace_with"
            ),
            pytest.param({}, id="missing_all"),
        ],
    )
    def test_edit_missing_required_arguments(self, kwargs):
        """Test that execute returns an error if required arguments are missing."""
        result = self.tool.execute(**kwargs)
        assert result["status"] == "error"
        assert (
            "The 'path', 'find_string', and 'replace_with' arguments are all required."
            in result["message"]
        )


class TestListDirectoryTool:
    def setup_method(self):
        self.tool = ListDirectoryTool()

    def test_list_directory_simple(self, work_dir: Path):
        """Test listing the contents of the current directory."""
        (work_dir / "file1.txt").touch()
        (work_dir / "subdir").mkdir()

        result = self.tool.execute()  # Defaults to path='.'
        assert result["status"] == "success"
        assert sorted(result["contents"]) == ["file1.txt", "subdir/"]

    def test_list_subdirectory(self, work_dir: Path):
        """Test listing the contents of a specific subdirectory."""
        (work_dir / "subdir").mkdir()
        (work_dir / "subdir" / "file2.py").touch()

        result = self.tool.execute(path="subdir")
        assert result["status"] == "success"
        assert result["contents"] == ["subdir/file2.py"]

    def test_list_with_hidden_files(self, work_dir: Path):
        """Test that hidden files are excluded by default but included when requested."""
        (work_dir / ".hidden_file").touch()
        (work_dir / "visible_file.txt").touch()

        # Default behavior (hidden files excluded)
        result1 = self.tool.execute()
        assert result1["status"] == "success"
        assert ".hidden_file" not in result1["contents"]
        assert "visible_file.txt" in result1["contents"]

        # With show_hidden=True
        result2 = self.tool.execute(show_hidden=True)
        assert result2["status"] == "success"
        assert ".hidden_file" in result2["contents"]
        assert "visible_file.txt" in result2["contents"]

    def test_list_recursive(self, work_dir: Path):
        """Test recursive directory listing."""
        (work_dir / "file1.txt").touch()
        (work_dir / "subdir").mkdir()
        (work_dir / "subdir" / "file2.py").touch()
        (work_dir / "subdir" / ".hidden").touch()

        result = self.tool.execute(recursive=True)
        assert result["status"] == "success"
        expected = sorted(["file1.txt", "subdir/", "subdir/file2.py"])
        assert sorted(result["contents"]) == expected

    def test_list_path_is_not_a_directory(self, work_dir: Path):
        """Test that listing a file path returns an error."""
        (work_dir / "a_file.txt").touch()
        result = self.tool.execute(path="a_file.txt")
        assert result["status"] == "error"
        assert "Path is not a directory" in result["message"]

    def test_list_unsafe_path_fails(self, work_dir: Path):
        """Test that listing an unsafe path fails."""
        result = self.tool.execute(path="../")
        assert result["status"] == "error"
        assert "is outside the safe working directory" in result["message"]
