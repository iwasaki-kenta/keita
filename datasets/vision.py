import errno
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Omniglot(Dataset):
    URLS = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'

    def __init__(self, root='data/omniglot', transform=None, target_transform=None, download=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        if download: self.download()

        assert self._check_exists(), 'Dataset not found. You can use download=True to download it'

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.classes = index_classes(self.all_items)

    def __getitem__(self, index):
        class_name = list(self.all_items.keys())[index]
        images = self.all_items[class_name]
        target = self.classes[class_name]

        sampled_index = random.randint(0, len(images) - 1)
        image_path = os.path.join(images[sampled_index][1], images[sampled_index][0])
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.classes)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.URLS:
            print('Downloading %s.' % url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)

            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    classes = {}
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("png"):
                path = root.split(os.pathsep)
                class_name = os.path.join(path[len(path) - 2], path[len(path) - 1])
                if not class_name in classes:
                    classes[class_name] = []
                classes[class_name].append((file, root))
    return classes


def index_classes(classes):
    indices = {}
    for index, cls in enumerate(classes.keys()):
        indices[cls] = index
    return indices


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    image_transforms = transforms.Compose([
        transforms.Scale(28),
        transforms.ToTensor()
    ])
    dataset = Omniglot(transform=image_transforms)

    iterator = DataLoader(dataset, 5, shuffle=True, drop_last=True)

    batch = next(iter(iterator))
    images, labels = batch

    print("A normal batch looks like %s w/ labels as %s. " % (str(images.size()), str(labels.size())))
    print("The dataset contains %d classes. " % len(dataset))
