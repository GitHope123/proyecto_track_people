import os
import glob

def clean_orphans(images_dir, labels_dir):
    print(f"Scanning {images_dir} and {labels_dir}...")
    
    # Supported image extensions
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Get all images
    images = []
    for ext in img_exts:
        images.extend(glob.glob(os.path.join(images_dir, f'*{ext}')))
        images.extend(glob.glob(os.path.join(images_dir, f'*{ext.upper()}')))
    
    # Get all labels
    labels = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    image_names = {os.path.splitext(os.path.basename(x))[0] for x in images}
    label_names = {os.path.splitext(os.path.basename(x))[0] for x in labels}
    
    # Images without labels
    orphaned_images = image_names - label_names
    # Labels without images
    orphaned_labels = label_names - image_names
    
    print(f"Found {len(orphaned_images)} images without labels.")
    print(f"Found {len(orphaned_labels)} labels without images.")
    
    # Delete orphaned images
    for img_name in orphaned_images:
        found = False
        for ext in img_exts:
            p = os.path.join(images_dir, img_name + ext)
            if os.path.exists(p):
                print(f"Deleting orphan image: {p}")
                os.remove(p)
                found = True
                break
            p = os.path.join(images_dir, img_name + ext.upper())
            if os.path.exists(p):
                print(f"Deleting orphan image: {p}")
                os.remove(p)
                found = True
                break

    # Delete orphaned labels
    for lbl_name in orphaned_labels:
        p = os.path.join(labels_dir, lbl_name + '.txt')
        if os.path.exists(p):
            print(f"Deleting orphan label: {p}")
            os.remove(p)

if __name__ == "__main__":
    base_dir = "/home/hugo/proyecto_track_people/notebooks/overhead"
    
    # Pairs of (images_dir, labels_dir)
    dirs = [
        ("train/images", "train/labels"),
        ("valid/images", "valid/labels"),
        ("test/images", "test/labels")
    ]
    
    for img_sub, lbl_sub in dirs:
        img_path = os.path.join(base_dir, img_sub)
        lbl_path = os.path.join(base_dir, lbl_sub)
        
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            clean_orphans(img_path, lbl_path)
        else:
            print(f"Skipping pair {img_sub} - {lbl_sub} (one or both not found)")
