import os
import sys
import argparse

# Add src to python path so we can import cv_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cv_pipeline.face_recognizer import FaceDB
from cv_pipeline.config import config

def enroll(db_path: str, name: str, image_path: str):
    db = FaceDB(db_path)
    try:
        db.enroll(name, image_path)
        print(f"Successfully enrolled '{name}'")
    except Exception as e:
        print(f"Error: {e}")

def list_faces(db_path: str):
    db = FaceDB(db_path).load()
    names = db.get("names", [])
    if not names:
        print("No faces enrolled.")
    else:
        print(f"Enrolled faces ({len(names)}):")
        for i, name in enumerate(set(names)):
            print(f"  {i+1}. {name}")

def delete_face(db_path: str, name: str):
    db = FaceDB(db_path)
    if db.delete(name):
        print(f"Successfully deleted '{name}'")
    else:
        print(f"Name '{name}' not found in database.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesto-Drive Face Enrollment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Enroll
    parser_enroll = subparsers.add_parser("enroll", help="Enroll a new face")
    parser_enroll.add_argument("name", help="Name of the person")
    parser_enroll.add_argument("image", help="Path to the image file")
    
    # List
    parser_list = subparsers.add_parser("list", help="List enrolled faces")
    
    # Delete
    parser_delete = subparsers.add_parser("delete", help="Delete an enrolled face")
    parser_delete.add_argument("name", help="Name of the person to delete")
    
    args = parser.parse_args()
    
    db_path = config.face_db_path
    
    if args.command == "enroll":
        enroll(db_path, args.name, args.image)
    elif args.command == "list":
        list_faces(db_path)
    elif args.command == "delete":
        delete_face(db_path, args.name)
