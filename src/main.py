#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maa Pipeline JSON Formatter
Format MaaFramework Pipeline JSON files (Batch Processing Mode)
Supports format and diff modes with flexible path handling
"""

import json
import re
import sys
import os
import difflib
import argparse
from typing import Any, Dict, List, Union, Tuple, Optional
from pathlib import Path
from datetime import datetime


# ============================================================================
# Configuration Management (保持不变)
# ============================================================================

DEFAULT_CONFIG = {
    "version": "1.0",
    "indent": {
        "style": "space",
        "width": 4
    },
    "formatting": {
        "simple_array_threshold": 50,
        "coordinate_fields": [
            "roi", "roi_offset", "target", "target_offset",
            "begin", "begin_offset", "end", "end_offset",
            "lower", "upper"
        ],
        "control_flow_fields": [
            "next", "interrupt", "on_error", "template"
        ],
        "always_multiline_fields": [
            "custom_action_param", "custom_param",
            "parameters", "params", "options", "config"
        ]
    },
    "file_handling": {
        "preserve_comments": True,
        "output_suffix": ".formatted",
        "encoding": "utf-8",
        "newline": "LF",
        "diff_context_lines": 3,
        "diff_suffix": ".diff"
    }
}


def find_config_file(start_path: Path) -> Optional[Path]:
    """Find .maapipeline-format config file by searching upward"""
    current = start_path.resolve()
    
    while current != current.parent:
        config_file = current / ".maapipeline-format"
        if config_file.exists() and config_file.is_file():
            return config_file
        current = current.parent
    
    home_config = Path.home() / ".maapipeline-format"
    if home_config.exists() and home_config.is_file():
        return home_config
    
    return None


def load_config(config_path: Optional[Path] = None, search_from: Optional[Path] = None) -> Dict:
    """Load configuration from file or use defaults"""
    if config_path is not None:
        if not config_path.exists():
            print(f"[WARN] Config file not found: {config_path}")
            print(f"[INFO] Using default configuration")
            return DEFAULT_CONFIG.copy()
    else:
        if search_from is None:
            search_from = Path.cwd()
        config_path = find_config_file(search_from)
    
    if config_path is None:
        print(f"[INFO] No config file found, using default configuration")
        return DEFAULT_CONFIG.copy()
    
    try:
        print(f"[INFO] Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        
        config = _merge_config(DEFAULT_CONFIG, user_config)
        
        valid, errors = _validate_config(config)
        if not valid:
            print(f"[WARN] Config validation failed:")
            for error in errors:
                print(f"       - {error}")
            print(f"[INFO] Using default configuration")
            return DEFAULT_CONFIG.copy()
        
        print(f"[OK] Config loaded successfully")
        return config
        
    except json.JSONDecodeError as e:
        print(f"[FAIL] Failed to parse config file: {e}")
        print(f"[INFO] Using default configuration")
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"[FAIL] Failed to load config: {e}")
        print(f"[INFO] Using default configuration")
        return DEFAULT_CONFIG.copy()


def _merge_config(base: Dict, override: Dict) -> Dict:
    """Deep merge two config dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def _validate_config(config: Dict) -> Tuple[bool, List[str]]:
    """Validate configuration dictionary"""
    errors = []
    
    if "version" not in config:
        errors.append("Missing 'version' field")
    elif config["version"] not in ["1.0"]:
        errors.append(f"Unsupported version: {config['version']}")
    
    if "indent" in config:
        indent_cfg = config["indent"]
        if "style" in indent_cfg and indent_cfg["style"] not in ["space", "tab"]:
            errors.append(f"Invalid indent style: {indent_cfg['style']}")
        if "width" in indent_cfg:
            if not isinstance(indent_cfg["width"], int) or indent_cfg["width"] < 1:
                errors.append(f"Invalid indent width: {indent_cfg['width']}")
    
    if "formatting" in config:
        fmt_cfg = config["formatting"]
        if "simple_array_threshold" in fmt_cfg:
            threshold = fmt_cfg["simple_array_threshold"]
            if not isinstance(threshold, int) or threshold < 0:
                errors.append(f"Invalid simple_array_threshold: {threshold}")
    
    if "file_handling" in config:
        fh_cfg = config["file_handling"]
        if "diff_context_lines" in fh_cfg:
            context = fh_cfg["diff_context_lines"]
            if not isinstance(context, int) or context < 0:
                errors.append(f"Invalid diff_context_lines: {context}")
    
    return (len(errors) == 0, errors)


# ============================================================================
# Maa Pipeline Formatter (保持不变)
# ============================================================================

class MaaPipelineFormatter:
    """MaaFramework Pipeline JSON Formatter"""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = DEFAULT_CONFIG.copy()
        
        self.config = config
        
        indent_cfg = config["indent"]
        indent_char = "\t" if indent_cfg["style"] == "tab" else " "
        indent_width = indent_cfg["width"]
        self.indent = indent_char * indent_width
        
        fmt_cfg = config["formatting"]
        self.coordinate_fields = set(fmt_cfg["coordinate_fields"])
        self.control_flow_fields = set(fmt_cfg["control_flow_fields"])
        self.always_multiline_fields = set(fmt_cfg["always_multiline_fields"])
        self.simple_array_threshold = fmt_cfg["simple_array_threshold"]
        
        fh_cfg = config["file_handling"]
        self.preserve_comments = fh_cfg["preserve_comments"]
        self.output_suffix = fh_cfg["output_suffix"]
        self.encoding = fh_cfg["encoding"]
        self.newline = "\r\n" if fh_cfg["newline"] == "CRLF" else "\n"
        self.diff_context_lines = fh_cfg["diff_context_lines"]
        self.diff_suffix = fh_cfg["diff_suffix"]
    
    def _is_simple_value(self, value: Any) -> bool:
        return isinstance(value, (str, int, float, bool, type(None)))
    
    def _is_coordinate_array(self, key: str, value: Any) -> bool:
        if key not in self.coordinate_fields:
            return False
        if not isinstance(value, list):
            return False
        return all(isinstance(v, (int, float)) for v in value)
    
    def _should_inline_array(self, key: str, value: List) -> bool:
        if not value:
            return True
        
        if self._is_coordinate_array(key, value):
            return True
        
        if key in self.control_flow_fields:
            return False
        
        if all(self._is_simple_value(v) for v in value):
            inline_str = json.dumps(value, ensure_ascii=False)
            return len(inline_str) <= self.simple_array_threshold
        
        return False
    
    def _should_inline_object(self, key: str, value: Dict) -> bool:
        if not value:
            return True
        
        if key in self.always_multiline_fields:
            return False
        
        if all(self._is_simple_value(v) for v in value.values()):
            inline_str = json.dumps(value, ensure_ascii=False)
            return len(inline_str) <= self.simple_array_threshold
        
        return False
    
    def _format_value(self, key: str, value: Any, indent_level: int, parent_is_root: bool = False) -> str:
        if self._is_simple_value(value):
            return json.dumps(value, ensure_ascii=False)
        
        if isinstance(value, list):
            if self._should_inline_array(key, value):
                return json.dumps(value, ensure_ascii=False)
            else:
                lines = ["["]
                for i, item in enumerate(value):
                    item_str = self._format_value("", item, indent_level + 1)
                    comma = "," if i < len(value) - 1 else ""
                    lines.append(f"{self.indent * (indent_level + 1)}{item_str}{comma}")
                lines.append(f"{self.indent * indent_level}]")
                return "\n".join(lines)
        
        if isinstance(value, dict):
            if parent_is_root:
                return self._format_object(value, indent_level, is_root=False)
            
            if self._should_inline_object(key, value):
                return json.dumps(value, ensure_ascii=False)
            else:
                return self._format_object(value, indent_level, is_root=False)
        
        return json.dumps(value, ensure_ascii=False)
    
    def _format_object(self, obj: Dict, indent_level: int, is_root: bool = False) -> str:
        if not obj:
            return "{}"
        
        lines = ["{"]
        items = list(obj.items())
        
        for i, (key, value) in enumerate(items):
            key_str = json.dumps(key, ensure_ascii=False)
            value_str = self._format_value(key, value, indent_level + 1, parent_is_root=is_root)
            comma = "," if i < len(items) - 1 else ""
            
            if "\n" in value_str:
                lines.append(f"{self.indent * (indent_level + 1)}{key_str}: {value_str}{comma}")
            else:
                lines.append(f"{self.indent * (indent_level + 1)}{key_str}: {value_str}{comma}")
        
        lines.append(f"{self.indent * indent_level}}}")
        return "\n".join(lines)
    
    def _preserve_comments(self, original_text: str, formatted_text: str) -> str:
        original_lines = original_text.split('\n')
        formatted_lines = formatted_text.split('\n')
        
        comment_map = {}
        current_comments = []
        
        for line in original_lines:
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                current_comments.append(line.rstrip())
            elif '"' in stripped and ':' in stripped:
                match = re.search(r'"([^"]+)"\s*:', stripped)
                if match:
                    node_name = match.group(1)
                    if current_comments:
                        comment_map[node_name] = current_comments.copy()
                        current_comments = []
        
        result_lines = []
        for line in formatted_lines:
            match = re.search(r'"([^"]+)"\s*:', line)
            if match:
                node_name = match.group(1)
                if node_name in comment_map:
                    indent = len(line) - len(line.lstrip())
                    for comment in comment_map[node_name]:
                        result_lines.append(' ' * indent + comment.lstrip())
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _remove_comments(self, text: str) -> str:
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    def format_text(self, text: str) -> Tuple[bool, str, str]:
        try:
            text_without_comments = self._remove_comments(text)
            
            try:
                data = json.loads(text_without_comments)
            except json.JSONDecodeError as e:
                return False, "", f"JSON parse error: {e}"
            
            formatted = self._format_object(data, 0, is_root=True)
            
            if self.preserve_comments:
                try:
                    final_text = self._preserve_comments(text, formatted)
                except Exception:
                    final_text = formatted
            else:
                final_text = formatted
            
            return True, final_text, ""
            
        except Exception as e:
            return False, "", str(e)
    
    def format_file(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> Tuple[bool, str]:
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        try:
            with open(input_path, 'r', encoding=self.encoding) as f:
                original_text = f.read()
            
            success, formatted_text, error_msg = self.format_text(original_text)
            
            if not success:
                return False, error_msg
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding=self.encoding, newline=self.newline) as f:
                f.write(formatted_text)
            
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def generate_diff(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> Tuple[bool, bool, str]:
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        try:
            with open(input_path, 'r', encoding=self.encoding) as f:
                original_text = f.read()
            
            success, formatted_text, error_msg = self.format_text(original_text)
            
            if not success:
                return False, False, error_msg
            
            original_lines = original_text.splitlines(keepends=True)
            formatted_lines = formatted_text.splitlines(keepends=True)
            
            diff_lines = list(difflib.unified_diff(
                original_lines,
                formatted_lines,
                fromfile=f"a/{input_path.name}",
                tofile=f"b/{input_path.name}",
                n=self.diff_context_lines,
                lineterm=''
            ))
            
            has_changes = len(diff_lines) > 0
            
            if has_changes:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding=self.encoding, newline=self.newline) as f:
                    f.write('\n'.join(diff_lines))
            
            return True, has_changes, ""
            
        except Exception as e:
            return False, False, str(e)


# ============================================================================
# Utility Functions
# ============================================================================

def scan_json_files(base_dir: Path) -> List[Path]:
    """Recursively scan for JSON files in directory"""
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("*.json"))


def format_size(size_bytes: int) -> str:
    """Format file size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Maa Pipeline JSON Formatter - Format MaaFramework pipeline JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Format files in-place (modify original files)
  python main.py /path/to/pipeline/tasks --in-place
  
  # Generate diff files only (to review changes)
  python main.py /path/to/pipeline/tasks --diff-only --diff-dir ./diffs
  
  # Format and generate diffs
  python main.py /path/to/pipeline/tasks --in-place --diff-dir ./diffs
  
  # Non-interactive mode (for CI/CD)
  python main.py /path/to/pipeline/tasks --in-place --yes
        """
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        nargs='?',
        help='Input directory containing JSON files (default: test/input/)'
    )
    
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Format files in-place (modify original files)'
    )
    
    parser.add_argument(
        '--diff-only',
        action='store_true',
        help='Only generate diff files without modifying originals'
    )
    
    parser.add_argument(
        '--diff-dir',
        type=str,
        help='Directory to save diff files (default: <input_path>/../diff/)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for formatted files (only used without --in-place)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to .maapipeline-format config file'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompts (for CI/CD)'
    )
    
    return parser.parse_args()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function: Batch formatting or diff generation"""
    print("=" * 70)
    print(" " * 10 + "Maa Pipeline JSON Formatter (Batch Mode)")
    print(" " * 5 + "MaaFramework Task Pipeline JSON Batch Formatting Tool")
    print("=" * 70)
    print()
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine input directory
    if args.input_path:
        input_dir = Path(args.input_path).resolve()
    else:
        script_dir = Path(__file__).parent.parent
        input_dir = script_dir / "test" / "input"
    
    if not input_dir.exists():
        print(f"[FAIL] Input directory not found: {input_dir}")
        return
    
    print(f"[DIR] Input directory: {input_dir}")
    
    # Determine operation mode
    if args.diff_only:
        mode = 'diff'
    elif args.in_place:
        mode = 'in-place'
    else:
        # Interactive mode selection
        if not sys.stdin.isatty() or args.yes:
            mode = 'diff'
            print(f"[INFO] Default mode: diff (non-interactive)")
        else:
            print("\nSelect operation mode:")
            print("  1. Format in-place - Modify original files")
            print("  2. Diff only       - Generate diff files only")
            print("  3. Both            - Format in-place and generate diffs")
            print()
            
            while True:
                try:
                    choice = input("Enter your choice (1/2/3): ").strip()
                    if choice == '1':
                        mode = 'in-place'
                        break
                    elif choice == '2':
                        mode = 'diff'
                        break
                    elif choice == '3':
                        mode = 'both'
                        break
                    else:
                        print("[WARN] Invalid choice, please enter 1, 2, or 3")
                except (KeyboardInterrupt, EOFError):
                    print("\n[FAIL] Operation cancelled")
                    return
    
    # Handle 'both' mode
    generate_diff = mode in ['diff', 'both']
    format_in_place = mode in ['in-place', 'both']
    
    # Determine diff directory
    if generate_diff:
        if args.diff_dir:
            diff_dir = Path(args.diff_dir).resolve()
        else:
            diff_dir = input_dir.parent / "diff"
        print(f"[DIR] Diff directory:  {diff_dir}")
    
    # Determine output directory (only for non-in-place mode)
    if not format_in_place and args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        print(f"[DIR] Output directory: {output_dir}")
    
    print()
    
    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path=config_path, search_from=input_dir)
    print()
    
    # Scan JSON files
    print("[SCAN] Scanning JSON files...")
    json_files = scan_json_files(input_dir)
    
    if not json_files:
        print(f"[WARN] No JSON files found in: {input_dir}")
        return
    
    print(f"[OK] Found {len(json_files)} JSON files")
    print()
    
    # Display file list (limit to first 10)
    print("[LIST] File list:")
    display_files = json_files[:10]
    for i, file in enumerate(display_files, 1):
        relative = file.relative_to(input_dir)
        size = format_size(file.stat().st_size)
        print(f"       {i:3d}. {relative} ({size})")
    
    if len(json_files) > 10:
        print(f"       ... and {len(json_files) - 10} more files")
    print()
    
    # Confirm processing
    if format_in_place and generate_diff:
        prompt = f"Format {len(json_files)} files in-place AND generate diffs? (y/n): "
    elif format_in_place:
        prompt = f"Format {len(json_files)} files in-place? (y/n): "
    else:
        prompt = f"Generate diff for {len(json_files)} files? (y/n): "
    
    if not sys.stdin.isatty() or args.yes:
        print(f"[INFO] Auto-confirming (--yes flag or non-interactive mode)")
        confirm = 'y'
    else:
        try:
            confirm = input(prompt).strip().lower()
            if confirm != 'y':
                print("[FAIL] Operation cancelled")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n[FAIL] Operation cancelled")
            return
    
    print()
    print("=" * 70)
    if format_in_place and generate_diff:
        print("[START] Formatting in-place and generating diffs...")
    elif format_in_place:
        print("[START] Formatting files in-place...")
    else:
        print("[START] Generating diff files...")
    print("=" * 70)
    print()
    
    # Batch processing
    formatter = MaaPipelineFormatter(config)
    
    success_count = 0
    fail_count = 0
    unchanged_count = 0
    failed_files = []
    
    start_time = datetime.now()
    
    for i, input_path in enumerate(json_files, 1):
        relative_path = input_path.relative_to(input_dir)
        
        print(f"[{i}/{len(json_files)}] Processing: {relative_path}")
        
        file_has_error = False
        
        # Generate diff if requested
        if generate_diff:
            diff_output_path = diff_dir / relative_path.parent / f"{relative_path.name}.diff"
            
            success, has_changes, error_msg = formatter.generate_diff(input_path, diff_output_path)
            
            if success:
                if has_changes:
                    diff_size = format_size(diff_output_path.stat().st_size)
                    print(f"            [DIFF] Generated -> {diff_output_path.relative_to(diff_dir)} ({diff_size})")
                else:
                    unchanged_count += 1
                    print(f"            [INFO] No formatting changes needed")
            else:
                file_has_error = True
                failed_files.append((relative_path, error_msg))
                print(f"            [FAIL] Diff generation failed: {error_msg}")
        
        # Format in-place if requested
        if format_in_place and not file_has_error:
            success, error_msg = formatter.format_file(input_path, input_path)
            
            if success:
                success_count += 1
                file_size = format_size(input_path.stat().st_size)
                print(f"            [OK] Formatted in-place ({file_size})")
            else:
                fail_count += 1
                failed_files.append((relative_path, error_msg))
                print(f"            [FAIL] Formatting failed: {error_msg}")
        elif not file_has_error and not generate_diff:
            # Only count as success if we're not generating diffs
            success_count += 1
        
        print()
    
    # Statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("=" * 70)
    print("[STAT] Processing completed! Statistics:")
    print("=" * 70)
    
    if format_in_place:
        print(f"[OK]   Files formatted: {success_count}")
    if generate_diff:
        print(f"[OK]   Files with changes: {len(json_files) - unchanged_count - fail_count}")
        print(f"[INFO] Files unchanged:    {unchanged_count}")
    print(f"[FAIL] Failed:             {fail_count}")
    print(f"[TIME] Elapsed:            {duration:.2f} seconds")
    
    if generate_diff:
        print(f"[DIR]  Diffs saved to:     {diff_dir}")
    
    print()
    
    # Failed files details
    if failed_files:
        print("[FAIL] Failed files details:")
        for file, error in failed_files:
            print(f"       - {file}")
            print(f"         Error: {error}")
        print()
    
    print("=" * 70)
    print("Thank you for using Maa Pipeline Formatter!")
    print("=" * 70)


if __name__ == "__main__":
    main()