from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import json
import networkx as nx


from extractors.VladExtractor import VLADExtractor
from extractors.DinoExtractor import DINOv2Extractor

CACHE_DIR = "cache"
IMAGE_DIR = "data/images/"
DATA_INFO_PATH = "data/data_info.json"

# Graph construction
TEMPORAL_WEIGHT = 1.0       # edge weight for consecutive frames
VISUAL_WEIGHT_BASE = 2.0    # base weight for visual shortcut edges
VISUAL_WEIGHT_SCALE = 3.0   # weight += scale * vlad_distance
MIN_SHORTCUT_GAP = 50       # minimum trajectory index gap for shortcuts

class GraphPlayer(Player):

    def __init__(self, extractor, subsample_rate: int = 5, top_k_shortcuts: int = 30, **kwargs):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super().__init__()

        self.subsample_rate = subsample_rate
        self.top_k_shortcuts = top_k_shortcuts

        # Load trajectory data
        self.motion_frames = []
        self.file_list = []
        if os.path.exists(DATA_INFO_PATH):
            with open(DATA_INFO_PATH) as f:
                raw = json.load(f)
            pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
            all_motion = [
                {'step': d['step'], 'image': d['image'], 'action': d['action'][0]}
                for d in raw
                if len(d['action']) == 1 and d['action'][0] in pure
            ]
            self.motion_frames = all_motion[::subsample_rate]
            self.file_list = [m['image'] for m in self.motion_frames]
            print(f"Frames: {len(all_motion)} total, "
                  f"{len(self.motion_frames)} after {subsample_rate}x subsample")

        if extractor == "VLAD":
            self.extractor = VLADExtractor(
                file_list=self.file_list, 
                img_dir=IMAGE_DIR, 
                cache_dir=CACHE_DIR, 
                subsample_rate=self.subsample_rate, 
                **kwargs
            )
        elif extractor == "DINO":
            self.extractor = DINOv2Extractor(
                file_list=self.file_list, 
                img_dir=IMAGE_DIR, 
                cache_dir=CACHE_DIR, 
                subsample_rate=self.subsample_rate
            )
        else:
            raise TypeError("Not a valid extractor type")

        self.database = None
        self.G = None
        self.goal_node = None

    # --- Game engine hooks ---
    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption("KeyboardPlayer:fpv")

        if self._state and self._state[1] == Phase.NAVIGATION:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                self.display_next_best_view()

        rgb = fpv[:, :, ::-1]
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], 'RGB')
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def set_target_images(self, images):
        super().set_target_images(images)
        self.show_target_images()

    def pre_navigation(self):
        super().pre_navigation()
        self._build_database()
        self._build_graph()
        self._setup_goal()

    # --- VLAD database ---
    def _build_database(self):
        """Compute VLAD database (skips if already done)."""
        if self.database is not None:
            print("Database already computed, skipping.")
            return
        self.database = self.extractor.extract_batch()
        print(f"Database: {self.database.shape}")

    # --- Navigation graph ---
    def _build_graph(self):
        """Build graph with temporal + visual shortcut edges."""
        if self.G is not None:
            print("Graph already built, skipping.")
            return

        n = len(self.database)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        # Temporal edges (consecutive frames)
        for i in range(n - 1):
            self.G.add_edge(i, i + 1, weight=TEMPORAL_WEIGHT, edge_type="temporal")

        # Visual shortcut edges: global top-K most similar pairs
        print("Computing similarity matrix...")
        sim = self.database @ self.database.T
        np.fill_diagonal(sim, -2)

        # Mask nearby pairs + lower triangle
        for i in range(n):
            lo = max(0, i - MIN_SHORTCUT_GAP)
            hi = min(n, i + MIN_SHORTCUT_GAP + 1)
            sim[i, lo:hi] = -2
        sim[~np.triu(np.ones((n, n), dtype=bool), k=1)] = -2

        # Extract top-K
        flat = sim.ravel()
        top_k = self.top_k_shortcuts
        top_idx = np.argpartition(flat, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        dists = []
        print(f"Top-{top_k} shortcuts (min_gap={MIN_SHORTCUT_GAP}):")
        for rank, fi in enumerate(top_idx):
            i, j = divmod(int(fi), n)
            s = float(flat[fi])
            d = float(np.sqrt(max(0, 2 - 2 * s)))
            self.G.add_edge(i, j,
                            weight=VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * d,
                            edge_type="visual")
            dists.append(d)
            if rank < 5:
                print(f"  #{rank+1}: {i}<->{j} gap={abs(j-i)} d={d:.4f}")

        kd = np.array(dists)
        print(f"  {top_k} visual edges, dist: [{kd.min():.3f}, {kd.max():.3f}]")
        print(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    # --- Goal ---

    def _setup_goal(self):
        """Set goal node from front-view target image."""
        if self.goal_node is not None:
            print("Goal already set, skipping.")
            return
        targets = self.get_target_images()
        if not targets:
            return
        sims = self.database @ self.extractor.extract(targets[0])
        self.goal_node = int(np.argmax(sims))
        d = float(np.sqrt(max(0, 2 - 2 * sims[self.goal_node])))
        print(f"Goal: node {self.goal_node} (d={d:.4f})")

    # --- Helpers ---
    def _load_img(self, idx: int) -> np.ndarray | None:
        """Load image by database index."""
        if 0 <= idx < len(self.file_list):
            return cv2.imread(os.path.join(IMAGE_DIR, self.file_list[idx]))
        return None

    def _get_current_node(self) -> int:
        """Find best-matching database node for current FPV."""
        feat = self.extractor.extract(self.fpv)
        return int(np.argmax(self.database @ feat))

    def _get_path(self, start: int) -> list[int]:
        """Shortest path from start to goal_node."""
        try:
            return nx.shortest_path(self.G, start, self.goal_node, weight="weight")
        except nx.NetworkXNoPath:
            return [start]

    def _edge_action(self, a: int, b: int) -> str:
        """Get the action label for traversing edge a->b."""
        REVERSE = {'FORWARD': 'BACKWARD', 'BACKWARD': 'FORWARD',
                    'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if b == a + 1 and a < len(self.motion_frames):
            return self.motion_frames[a]['action']
        elif b == a - 1 and b < len(self.motion_frames):
            return REVERSE.get(self.motion_frames[b]['action'], '?')
        return '?'

    # --- Display ---
    def show_target_images(self):
        targets = self.get_target_images()
        if not targets:
            return
        top = cv2.hconcat(targets[:2])
        bot = cv2.hconcat(targets[2:])
        img = cv2.vconcat([top, bot])
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
        cv2.line(img, (0, h // 2), (w, h // 2), (0, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, pos in [('Front', (10, 25)), ('Right', (w//2+10, 25)),
                           ('Back', (10, h//2+25)), ('Left', (w//2+10, h//2+25))]:
            cv2.putText(img, label, pos, font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Target Images', img)
        cv2.waitKey(1)

    def display_next_best_view(self):
        """
        Navigation panel:
            Info bar: current node | goal | hops | next action
            Row 1:    [Live FPV] [Best match] [Target (front)]
            Row 2:    Path preview (next 5 nodes)
        """
        ACT = {'FORWARD': 'FWD', 'BACKWARD': 'BACK', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT'}
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA
        TW, TH = 260, 195          # main thumbnails
        PW, PH = TW * 3 // 5, TH * 3 // 5   # path preview thumbnails
        N_PREVIEW = 10

        # Localize & plan
        cur = self._get_current_node()
        cur_sim = float(self.database[cur] @ self.extractor.extract(self.fpv))
        cur_d = float(np.sqrt(max(0, 2 - 2 * cur_sim)))
        path = self._get_path(cur)
        hops = len(path) - 1

        # Analyze edges
        edge_info = []
        for a, b in zip(path[:-1], path[1:]):
            et = self.G[a][b].get("edge_type", "temporal")
            if et == "temporal":
                act = ACT.get(self._edge_action(a, b), '?')
                edge_info.append(("seq", act, b == a + 1))
            else:
                edge_info.append(("vis", None, None))
        t_steps = sum(1 for e in edge_info if e[0] == "seq")
        v_jumps = len(edge_info) - t_steps

        if edge_info:
            etype, act, _ = edge_info[0]
            hint = act if etype == "seq" else "VISUAL JUMP"
        else:
            hint = "AT GOAL"
        near = hops <= 5

        # --- Info bar ---
        panel_w = TW * 3
        bar = np.zeros((40, panel_w, 3), dtype=np.uint8)
        bar[:] = (0, 0, 160) if near else (50, 35, 15)
        txt = (f"Node {cur} (d={cur_d:.3f})"
               f"  |  Goal {self.goal_node}"
               f"  |  {hops} hops ({t_steps}s+{v_jumps}v)"
               f"  |  >> {hint}")
        cv2.putText(bar, txt, (8, 27), FONT, 0.48, (255, 255, 255), 1, AA)
        if near:
            cv2.putText(bar, "NEAR TARGET — SPACE",
                        (panel_w - 220, 27), FONT, 0.48, (0, 255, 255), 1, AA)

        # --- Row 1: [FPV] [Match] [Target] ---
        def thumb(img, label, color, extra=None):
            t = cv2.resize(img, (TW, TH))
            cv2.rectangle(t, (0, 0), (TW-1, TH-1), color, 2)
            cv2.putText(t, label, (6, 22), FONT, 0.55, color, 1, AA)
            if extra:
                cv2.putText(t, extra, (6, 44), FONT, 0.45, (200, 200, 200), 1, AA)
            return t

        fpv_t = thumb(self.fpv, "Live FPV", (255, 255, 255))
        match_img = self._load_img(cur)
        if match_img is None:
            match_img = np.zeros((TH, TW, 3), dtype=np.uint8)
        match_t = thumb(match_img, f"Match: node {cur}", (0, 255, 0), f"d={cur_d:.3f}")
        targets = self.get_target_images()
        tgt = targets[0] if targets else np.zeros((TH, TW, 3), dtype=np.uint8)
        tgt_t = thumb(tgt, "Target (front)", (0, 140, 255))
        row1 = cv2.hconcat([fpv_t, match_t, tgt_t])

        # --- Row 2: path preview ---
        # --- Row 2 & 3: Path Preview (Split into two rows) ---
        preview = path[1:1 + N_PREVIEW]
        all_cells = []
        for p in range(N_PREVIEW):
            if p < len(preview):
                img = self._load_img(preview[p])
                if img is None:
                    img = np.zeros((PH, PW, 3), dtype=np.uint8)
                img = cv2.resize(img, (PW, PH))
                
                # --- RE-ADDED TEXT LOGIC ---
                etype, act, is_fwd = edge_info[p]
                if etype == "seq":
                    lbl = f"{'>' if is_fwd else '<'} {act}"
                    clr = (200, 200, 0) # Cyan-ish
                else:
                    lbl = "~ VISUAL"
                    clr = (200, 100, 255) # Pink-ish
                
                cv2.rectangle(img, (0, 0), (PW-1, PH-1), clr, 1)
                cv2.putText(img, f"+{p+1} node {preview[p]}", (4, 16),
                            FONT, 0.38, (255, 255, 255), 1, AA)
                cv2.putText(img, lbl, (4, 34), FONT, 0.38, clr, 1, AA)
                # ---------------------------
            else:
                # Blank placeholder for empty slots
                img = np.zeros((PH, PW, 3), dtype=np.uint8)
            all_cells.append(img)

        # Split cells: 0-4 on Row 2, 5-9 on Row 3
        row2 = cv2.hconcat(all_cells[:5])
        row3 = cv2.hconcat(all_cells[5:10])

        # --- Final Assembly (Ensuring everything is 780px wide) ---
        panel_w = 780 # Standard width for 3 large thumbs or 5 small thumbs
        
        def pad_to_width(image, target_w):
            if image.shape[1] < target_w:
                p = np.zeros((image.shape[0], target_w - image.shape[1], 3), dtype=np.uint8)
                return cv2.hconcat([image, p])
            return image

        row1 = pad_to_width(row1, panel_w)
        row2 = pad_to_width(row2, panel_w)
        row3 = pad_to_width(row3, panel_w)
        
        # Ensure bar matches the width
        bar_resized = cv2.resize(bar, (panel_w, bar.shape[0]))

        panel = cv2.vconcat([bar_resized, row1, row2, row3])
        
        cv2.imshow("Navigation", panel)
        cv2.waitKey(1)
        print(f"Node {cur} -> Goal {self.goal_node} | "
              f"{hops} hops ({t_steps}s+{v_jumps}v) | >> {hint}")