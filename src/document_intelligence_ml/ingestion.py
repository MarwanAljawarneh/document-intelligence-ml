from pathlib import Path


def read_document(file_path: str) -> str:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".txt":
        raise ValueError("Only .txt files are supported")

    return path.read_text(encoding="utf-8")


def read_documents(folder_path: str) -> list[str]:
    folder = Path(folder_path)

    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    documents = []

    for file in sorted(folder.iterdir()):
        if file.is_file() and file.suffix.lower() == ".txt":
            text = read_document(str(file))
            documents.append(text)

    return documents
