import numpy as np
import cv2
import pickle
import os

from tqdm import tqdm
from sklearn.cluster import KMeans

class VLADExtractor:
    """RootSIFT + VLAD with intra-normalization and power normalization."""

    def __init__(self, file_list, img_dir="data/images/", cache_dir="cache", subsample_rate=5, n_clusters: int = 128):
        
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.file_list = file_list
        self.subsample_rate = subsample_rate

        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create()
        self.codebook = None
        self._sift_cache: dict[str, np.ndarray] = {}

        self.load_sift_cache()
        self.build_vocabulary()

    @property
    def dim(self) -> int:
        return self.n_clusters * 128

    # --- Internal helpers ---

    @staticmethod
    def _root_sift(des: np.ndarray) -> np.ndarray:
        """L1-normalize then sqrt (Hellinger kernel approximation)."""
        des = des / np.sum(des, axis=1, keepdims=True)
        return np.sqrt(des)

    def _des_to_vlad(self, des: np.ndarray) -> np.ndarray:
        """Aggregate local descriptors into a single VLAD vector."""
        labels = self.codebook.predict(des)
        centers = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        vlad = np.zeros((k, des.shape[1]))
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                vlad[i] = np.sum(des[mask] - centers[i], axis=0)
                norm = np.linalg.norm(vlad[i])
                if norm > 0:
                    vlad[i] /= norm                     # intra-normalization
        vlad = vlad.ravel()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))   # power normalization
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm                                # L2 normalization
        return vlad

    # --- Public API ---

    def load_sift_cache(self):
        """Load or compute RootSIFT descriptors for all images."""
        cache_file = os.path.join(self.cache_dir, f"sift_ss{self.subsample_rate}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached SIFT from {cache_file}")
            with open(cache_file, "rb") as f:
                self._sift_cache = pickle.load(f)
            if all(fname in self._sift_cache for fname in self.file_list):
                return
            print("  Cache incomplete, re-extracting...")

        print(f"Extracting SIFT for {len(self.file_list)} images...")
        self._sift_cache = {}
        for fname in tqdm(self.file_list, desc="SIFT"):
            img = cv2.imread(os.path.join(self.img_dir, fname))
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                self._sift_cache[fname] = self._root_sift(des)
        with open(cache_file, "wb") as f:
            pickle.dump(self._sift_cache, f)
        print(f"  Saved {len(self._sift_cache)} descriptors -> {cache_file}")

    def build_vocabulary(self):
        """Fit KMeans codebook on cached SIFT descriptors."""
        cache_file = os.path.join(self.cache_dir, f"codebook_k{self.n_clusters}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached codebook from {cache_file}")
            with open(cache_file, "rb") as f:
                self.codebook = pickle.load(f)
            return

        all_des = np.vstack([self._sift_cache[f] for f in self.file_list
                             if f in self._sift_cache])
        print(f"Fitting KMeans (k={self.n_clusters}) on {len(all_des)} descriptors...")
        self.codebook = KMeans(
            n_clusters=self.n_clusters, init='k-means++',
            n_init=3, max_iter=300, tol=1e-4, verbose=1, random_state=42,
        ).fit(all_des)
        print(f"  {self.codebook.n_iter_} iters, inertia={self.codebook.inertia_:.0f}")
        with open(cache_file, "wb") as f:
            pickle.dump(self.codebook, f)

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Compute VLAD for a single BGR image."""
        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.dim)
        return self._des_to_vlad(self._root_sift(des))

    def extract_batch(self) -> np.ndarray:
        """Compute VLAD for all images using cached SIFT. Returns (N, dim)."""
        vectors = []
        for fname in tqdm(self.file_list, desc="VLAD"):
            if fname in self._sift_cache and len(self._sift_cache[fname]) > 0:
                vectors.append(self._des_to_vlad(self._sift_cache[fname]))
            else:
                vectors.append(np.zeros(self.dim))
        return np.array(vectors)