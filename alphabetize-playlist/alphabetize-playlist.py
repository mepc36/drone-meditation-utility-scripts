import os
from pathlib import Path

def alphabetize_playlist():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get all .txt files from input directory
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print("No .txt files found in ./input directory")
        return
    
    for input_file in txt_files:
        print(f"Processing: {input_file.name}")
        
        # Read the file - try different encodings
        lines = None
        for encoding in ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252']:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                print(f"  Successfully read with {encoding} encoding")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if lines is None:
            print(f"  Error: Could not read file with any supported encoding")
            continue
        
        if not lines:
            print(f"  Skipping empty file: {input_file.name}")
            continue
        
        # Check if first line is a header (contains tabs or common header keywords)
        has_header = False
        header_line = None
        if lines and ('\t' in lines[0] or 'Name' in lines[0] or 'Artist' in lines[0]):
            has_header = True
            header_line = lines[0]
            content_lines = lines[1:]
        else:
            content_lines = lines
        
        # Sort the content lines alphabetically
        sorted_lines = sorted(content_lines, key=lambda x: x.strip().lower())
        
        # Extract file paths from the Location column (last column if tab-separated)
        file_paths = []
        if has_header and '\t' in header_line:
            # Tab-separated file with header
            for line in sorted_lines:
                if line.strip():
                    columns = line.split('\t')
                    if columns:
                        # Get the last column (Location)
                        location = columns[-1].strip()
                        if location:
                            # Convert "Macintosh HD/Users/..." to "/Users/..."
                            if location.startswith("Macintosh HD"):
                                location = "/" + location[len("Macintosh HD/"):]
                            file_paths.append(location)
        else:
            # Plain text file - assume each line is a path
            file_paths = [line.strip() for line in sorted_lines if line.strip()]
        
        # Write m3u playlist
        output_file = output_dir / f"{input_file.stem}_sorted.m3u"
        write_m3u(file_paths, output_file)
        
        print(f"  Created playlist with {len(file_paths)} tracks: {output_file.name}")
    
    print("\nDone!")


def write_m3u(file_paths: list[str], output_path: Path) -> None:
    """Write m3u playlist file."""
    lines = ["#EXTM3U", *file_paths, ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    alphabetize_playlist()
