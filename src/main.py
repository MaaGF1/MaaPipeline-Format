#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maa Pipeline JSON Formatter
Format MaaFramework Pipeline JSON files (Batch Processing Mode)
Supports format and diff modes
"""

import json
import re
import sys
import os
import difflib
from typing import Any, Dict, List, Union, Tuple, Optional
from pathlib import Path
from datetime import datetime


# ============================================================================
# Configuration Management
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
    """
    Find .maapipeline-format config file by searching upward from start_path
    
    Search order:
    1. Current directory and parent directories (recursively upward)
    2. User home directory (~/.maapipeline-format)
    3. None (use default config)
    
    Args:
        start_path: Starting search path
    
    Returns:
        Path to config file, or None if not found
    """
    current = start_path.resolve()
    
    # Search upward from current directory
    while current != current.parent:
        config_file = current / ".maapipeline-format"
        if config_file.exists() and config_file.is_file():
            return config_file
        current = current.parent
    
    # Check user home directory
    home_config = Path.home() / ".maapipeline-format"
    if home_config.exists() and home_config.is_file():
        return home_config
    
    return None


def load_config(config_path: Optional[Path] = None, search_from: Optional[Path] = None) -> Dict:
    """
    Load configuration from file or use defaults
    
    Args:
        config_path: Explicit config file path (overrides search)
        search_from: Directory to start config file search from
    
    Returns:
        Configuration dictionary
    """
    # If explicit path provided, use it
    if config_path is not None:
        if not config_path.exists():
            print(f"[WARN] Config file not found: {config_path}")
            print(f"[INFO] Using default configuration")
            return DEFAULT_CONFIG.copy()
    else:
        # Search for config file
        if search_from is None:
            search_from = Path.cwd()
        config_path = find_config_file(search_from)
    
    if config_path is None:
        print(f"[INFO] No config file found, using default configuration")
        return DEFAULT_CONFIG.copy()
    
    # Load and parse config
    try:
        print(f"[INFO] Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        
        # Merge with defaults (user config overrides defaults)
        config = _merge_config(DEFAULT_CONFIG, user_config)
        
        # Validate config
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
    """
    Deep merge two config dictionaries (override takes precedence)
    
    Args:
        base: Base configuration
        override: Override configuration
    
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def _validate_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration to validate
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Version check
    if "version" not in config:
        errors.append("Missing 'version' field")
    elif config["version"] not in ["1.0"]:
        errors.append(f"Unsupported version: {config['version']}")
    
    # Indent validation
    if "indent" in config:
        indent_cfg = config["indent"]
        if "style" in indent_cfg and indent_cfg["style"] not in ["space", "tab"]:
            errors.append(f"Invalid indent style: {indent_cfg['style']}")
        if "width" in indent_cfg:
            if not isinstance(indent_cfg["width"], int) or indent_cfg["width"] < 1:
                errors.append(f"Invalid indent width: {indent_cfg['width']}")
    
    # Formatting validation
    if "formatting" in config:
        fmt_cfg = config["formatting"]
        if "simple_array_threshold" in fmt_cfg:
            threshold = fmt_cfg["simple_array_threshold"]
            if not isinstance(threshold, int) or threshold < 0:
                errors.append(f"Invalid simple_array_threshold: {threshold}")
    
    # File handling validation
    if "file_handling" in config:
        fh_cfg = config["file_handling"]
        if "diff_context_lines" in fh_cfg:
            context = fh_cfg["diff_context_lines"]
            if not isinstance(context, int) or context < 0:
                errors.append(f"Invalid diff_context_lines: {context}")
    
    return (len(errors) == 0, errors)


def save_default_config(output_path: Path) -> bool:
    """
    Save default configuration to file
    
    Args:
        output_path: Output file path
    
    Returns:
        Success flag
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[FAIL] Failed to save default config: {e}")
        return False


# ============================================================================
# Maa Pipeline Formatter
# ============================================================================

class MaaPipelineFormatter:
    """MaaFramework Pipeline JSON Formatter"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize formatter with configuration
        
        Args:
            config: Configuration dictionary (uses defaults if None)
        """
        if config is None:
            config = DEFAULT_CONFIG.copy()
        
        self.config = config
        
        # Parse indent settings
        indent_cfg = config["indent"]
        indent_char = "\t" if indent_cfg["style"] == "tab" else " "
        indent_width = indent_cfg["width"]
        self.indent = indent_char * indent_width
        
        # Parse formatting settings
        fmt_cfg = config["formatting"]
        self.coordinate_fields = set(fmt_cfg["coordinate_fields"])
        self.control_flow_fields = set(fmt_cfg["control_flow_fields"])
        self.always_multiline_fields = set(fmt_cfg["always_multiline_fields"])
        self.simple_array_threshold = fmt_cfg["simple_array_threshold"]
        
        # Parse file handling settings
        fh_cfg = config["file_handling"]
        self.preserve_comments = fh_cfg["preserve_comments"]
        self.output_suffix = fh_cfg["output_suffix"]
        self.encoding = fh_cfg["encoding"]
        self.newline = "\r\n" if fh_cfg["newline"] == "CRLF" else "\n"
        self.diff_context_lines = fh_cfg["diff_context_lines"]
        self.diff_suffix = fh_cfg["diff_suffix"]
    
    def _is_simple_value(self, value: Any) -> bool:
        """Check if value is simple (string/number/bool/null)"""
        return isinstance(value, (str, int, float, bool, type(None)))
    
    def _is_coordinate_array(self, key: str, value: Any) -> bool:
        """Check if array is a coordinate array (should stay inline)"""
        if key not in self.coordinate_fields:
            return False
        if not isinstance(value, list):
            return False
        return all(isinstance(v, (int, float)) for v in value)
    
    def _should_inline_array(self, key: str, value: List) -> bool:
        """Determine if array should be displayed inline"""
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
        """Determine if object should be displayed inline"""
        if not value:
            return True
        
        if key in self.always_multiline_fields:
            return False
        
        if all(self._is_simple_value(v) for v in value.values()):
            inline_str = json.dumps(value, ensure_ascii=False)
            return len(inline_str) <= self.simple_array_threshold
        
        return False
    
    def _format_value(self, key: str, value: Any, indent_level: int, parent_is_root: bool = False) -> str:
        """Format a single value"""
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
        """Format an object"""
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
        """Preserve comments from original text"""
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
        """Remove JSON5 comments"""
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    def format_text(self, text: str) -> Tuple[bool, str, str]:
        """
        Format JSON text
        
        Args:
            text: Original JSON text
        
        Returns:
            (success, formatted_text, error_message)
        """
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
        """
        Format JSON file
        
        Args:
            input_path: Input file path
            output_path: Output file path
        
        Returns:
            (success, error_message)
        """
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
        """
        Generate unified diff file
        
        Args:
            input_path: Input file path
            output_path: Output diff file path
        
        Returns:
            (success, has_changes, error_message)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        try:
            with open(input_path, 'r', encoding=self.encoding) as f:
                original_text = f.read()
            
            success, formatted_text, error_msg = self.format_text(original_text)
            
            if not success:
                return False, False, error_msg
            
            # Generate unified diff
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
            
            # Check if there are any changes
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


def get_output_path(input_path: Path, input_dir: Path, output_dir: Path, suffix: str) -> Path:
    """Calculate output path preserving directory structure"""
    relative_path = input_path.relative_to(input_dir)
    output_path = output_dir / relative_path
    
    if suffix.startswith('.'):
        # Extension-like suffix (e.g., .diff)
        output_path = output_path.parent / f"{output_path.name}{suffix}"
    else:
        # Stem suffix (e.g., .formatted)
        output_path = output_path.with_stem(f"{output_path.stem}{suffix}")
    
    return output_path


def format_size(size_bytes: int) -> str:
    """Format file size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def parse_mode() -> str:
    """
    Parse operation mode from command line, environment, or interactive input
    
    Priority:
    1. Command line argument: python main.py diff
    2. Environment variable: MAA_FORMAT_MODE=diff
    3. Interactive input (only if TTY available)
    4. Default: diff (for non-interactive environments)
    
    Returns:
        'format' or 'diff'
    """
    # 1. Command line arguments take precedence
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ['format', 'diff']:
            return mode
        else:
            print(f"[FAIL] Invalid mode: {mode}")
            print("       Valid modes: format, diff")
            print()
            print("Usage:")
            print("  python main.py format    # Format files and save to output/")
            print("  python main.py diff      # Generate diff files to diff/")
            print("  python main.py           # Interactive mode selection")
            sys.exit(1)
    
    # 2. Environment Variables (for CI/CD)
    if 'MAA_FORMAT_MODE' in os.environ:
        mode = os.environ['MAA_FORMAT_MODE'].lower()
        if mode in ['format', 'diff']:
            print(f"[INFO] Mode from environment variable: {mode}")
            return mode
        else:
            print(f"[WARN] Invalid MAA_FORMAT_MODE: {mode}, ignoring")
    
    # 3. Interactive selection (only when TTY is available)
    if sys.stdin.isatty():
        print("Select operation mode:")
        print("  1. Format - Format files and save to output/")
        print("  2. Diff   - Generate diff files only to diff/")
        print()
        
        while True:
            try:
                choice = input("Enter your choice (1/2): ").strip()
                if choice == '1':
                    return 'format'
                elif choice == '2':
                    return 'diff'
                else:
                    print("[WARN] Invalid choice, please enter 1 or 2")
            except KeyboardInterrupt:
                print("\n[FAIL] Operation cancelled")
                sys.exit(1)
            except EOFError:
                print("\n[FAIL] Operation cancelled")
                sys.exit(1)
    else:
        # 4. Non-interactive environment default
        print("[INFO] Non-interactive environment detected (no TTY)")
        print("[INFO] Using default mode: diff")
        return 'diff'


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
    
    # Initialize paths
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / "test" / "input"
    output_dir = script_dir / "test" / "output"
    diff_dir = script_dir / "test" / "diff"
    
    print(f"[DIR] Project root: {script_dir}")
    print(f"[IN]  Input dir:    {input_dir}")
    print()
    
    # Parse operation mode
    mode = parse_mode()
    print()
    
    # Display mode information
    if mode == 'format':
        print(f"[INFO] Mode: Format")
        print(f"[OUT] Output dir:   {output_dir}")
    else:
        print(f"[INFO] Mode: Diff")
        print(f"[OUT] Diff dir:     {diff_dir}")
    print()
    
    # Load configuration
    config = load_config(search_from=script_dir)
    print()
    
    # Check input directory
    if not input_dir.exists():
        print(f"[FAIL] Input directory not found: {input_dir}")
        print(f"       Please ensure correct directory structure")
        return
    
    # Scan JSON files
    print("[SCAN] Scanning JSON files...")
    json_files = scan_json_files(input_dir)
    
    if not json_files:
        print(f"[WARN] No JSON files found in: {input_dir}")
        return
    
    print(f"[OK] Found {len(json_files)} JSON files")
    print()
    
    # Display file list
    print("[LIST] File list:")
    for i, file in enumerate(json_files, 1):
        relative = file.relative_to(input_dir)
        size = format_size(file.stat().st_size)
        print(f"       {i:3d}. {relative} ({size})")
    print()
    
    # Confirm processing
    if mode == 'format':
        prompt = f"Format these {len(json_files)} files? (y/n): "
    else:
        prompt = f"Generate diff for these {len(json_files)} files? (y/n): "
    
    # Automatic confirmation in non-interactive environments
    if not sys.stdin.isatty():
        print(f"[INFO] Non-interactive mode: auto-confirming processing")
        confirm = 'y'
    else:
        try:
            confirm = input(prompt).strip().lower()
            if confirm != 'y':
                print("[FAIL] Operation cancelled")
                return
        except KeyboardInterrupt:
            print("\n[FAIL] Operation cancelled")
            return
        except EOFError:
            print("\n[FAIL] Operation cancelled")
            return
    
    print()
    print("=" * 70)
    if mode == 'format':
        print("[START] Starting batch formatting...")
    else:
        print("[START] Starting diff generation...")
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
        
        if mode == 'format':
            # Format mode
            output_path = get_output_path(
                input_path, 
                input_dir, 
                output_dir, 
                config["file_handling"]["output_suffix"]
            )
            
            success, error_msg = formatter.format_file(input_path, output_path)
            
            if success:
                success_count += 1
                output_size = format_size(output_path.stat().st_size)
                print(f"            [OK] -> {output_path.relative_to(output_dir)} ({output_size})")
            else:
                fail_count += 1
                failed_files.append((relative_path, error_msg))
                print(f"            [FAIL] {error_msg}")
        
        else:
            # Diff mode
            output_path = get_output_path(
                input_path,
                input_dir,
                diff_dir,
                config["file_handling"]["diff_suffix"]
            )
            
            success, has_changes, error_msg = formatter.generate_diff(input_path, output_path)
            
            if success:
                if has_changes:
                    success_count += 1
                    output_size = format_size(output_path.stat().st_size)
                    print(f"            [OK] Changes detected -> {output_path.relative_to(diff_dir)} ({output_size})")
                else:
                    unchanged_count += 1
                    print(f"            [INFO] No changes needed")
            else:
                fail_count += 1
                failed_files.append((relative_path, error_msg))
                print(f"            [FAIL] {error_msg}")
        
        print()
    
    # Statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("=" * 70)
    if mode == 'format':
        print("[STAT] Formatting completed! Statistics:")
    else:
        print("[STAT] Diff generation completed! Statistics:")
    print("=" * 70)
    
    if mode == 'format':
        print(f"[OK]   Success: {success_count} files formatted")
        print(f"[FAIL] Failed:  {fail_count} files")
    else:
        print(f"[OK]   Files with changes: {success_count}")
        print(f"[INFO] Files unchanged:    {unchanged_count}")
        print(f"[FAIL] Failed:             {fail_count}")
    
    print(f"[TIME] Elapsed: {duration:.2f} seconds")
    
    if mode == 'format':
        print(f"[DIR]  Output:  {output_dir}")
    else:
        print(f"[DIR]  Output:  {diff_dir}")
    
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