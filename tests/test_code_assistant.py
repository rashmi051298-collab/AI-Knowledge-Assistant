import unittest
from pathlib import Path

from code_assistant import should_index_file


class ShouldIndexFileTests(unittest.TestCase):
    def test_includes_laravel_source(self) -> None:
        self.assertTrue(should_index_file(Path("app/Http/Controllers/UserController.php")))
        self.assertTrue(should_index_file(Path("resources/views/home.blade.php")))

    def test_excludes_generated_and_sensitive_files(self) -> None:
        self.assertFalse(should_index_file(Path("vendor/laravel/framework/src/Router.php")))
        self.assertFalse(should_index_file(Path("node_modules/pkg/index.js")))
        self.assertFalse(should_index_file(Path(".env")))
        self.assertFalse(should_index_file(Path(".env.production")))
        self.assertFalse(should_index_file(Path("composer.lock")))
        self.assertFalse(should_index_file(Path("certificates/private.pem")))

    def test_ignores_unsupported_extensions(self) -> None:
        self.assertFalse(should_index_file(Path("public/logo.png")))


if __name__ == "__main__":
    unittest.main()
