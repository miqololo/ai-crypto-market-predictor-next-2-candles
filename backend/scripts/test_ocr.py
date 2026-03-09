#!/usr/bin/env python3
"""Simple script to test OCR functionality."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ocr_service import get_ocr_service


def test_ocr(image_path: str, languages: list = None, detail: int = 0):
    """
    Test OCR extraction on an image file.
    
    Args:
        image_path: Path to image file
        languages: List of language codes (e.g., ['en'])
        detail: 0 for text only, 1 for detailed results
    """
    print(f"Testing OCR on: {image_path}")
    print(f"Languages: {languages or ['en']}")
    print(f"Detail level: {detail}")
    print("-" * 50)
    
    try:
        ocr_service = get_ocr_service()
        
        # Extract text
        result = ocr_service.extract_from_file(
            image_path,
            languages=languages,
            detail=detail
        )
        
        # Print results
        print(f"\nEngine: {result['engine']}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"Word count: {result.get('word_count', 'N/A')}")
        print("\nExtracted Text:")
        print("=" * 50)
        print(result['text'])
        print("=" * 50)
        
        if detail == 1 and result.get('details'):
            print(f"\nFound {len(result['details'])} text regions:")
            for i, detail_item in enumerate(result['details'][:5], 1):  # Show first 5
                print(f"\n{i}. Text: {detail_item['text']}")
                print(f"   Confidence: {detail_item.get('confidence', 'N/A')}")
                print(f"   BBox: {detail_item.get('bbox', 'N/A')}")
            if len(result['details']) > 5:
                print(f"\n... and {len(result['details']) - 5} more regions")
        
        return result
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def list_languages():
    """List available languages for each OCR engine."""
    print("Available OCR Languages:")
    print("=" * 50)
    
    try:
        ocr_service = get_ocr_service()
        languages = ocr_service.get_available_languages()
        
        for engine, langs in languages.items():
            print(f"\n{engine.upper()}:")
            print(f"  Total: {len(langs)} languages")
            print(f"  Sample: {', '.join(langs[:10])}")
            if len(langs) > 10:
                print(f"  ... and {len(langs) - 10} more")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OCR functionality")
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to image file to process"
    )
    parser.add_argument(
        "-l", "--languages",
        nargs="+",
        default=["en"],
        help="Language codes (e.g., en es fr)"
    )
    parser.add_argument(
        "-d", "--detail",
        type=int,
        choices=[0, 1],
        default=0,
        help="Detail level: 0=text only, 1=with bounding boxes"
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List available languages and exit"
    )
    parser.add_argument(
        "-e", "--engine",
        choices=["easyocr", "tesseract"],
        help="OCR engine to use (default: auto-select)"
    )
    
    args = parser.parse_args()
    
    if args.list_languages:
        list_languages()
        sys.exit(0)
    
    if not args.image:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/test_ocr.py image.png")
        print("  python scripts/test_ocr.py image.png -l en es -d 1")
        print("  python scripts/test_ocr.py --list-languages")
        sys.exit(1)
    
    # Check if file exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: File not found: {args.image}")
        sys.exit(1)
    
    # Test OCR
    result = test_ocr(
        str(image_path),
        languages=args.languages,
        detail=args.detail
    )
    
    if result:
        print("\n✓ OCR extraction completed successfully!")
    else:
        print("\n✗ OCR extraction failed!")
        sys.exit(1)
