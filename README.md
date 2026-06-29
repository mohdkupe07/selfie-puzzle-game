# 🧩 Selfie Puzzle Game

A Computer Vision and Game Development project built using **Python, OpenCV, MediaPipe, NumPy, and Pygame**.

Selfie Puzzle Game is an interactive application that allows users to capture their selfie using **hand gestures**, automatically convert it into a shuffled **3×3 puzzle**, and solve it using a drag-and-drop game interface.

This project combines **Computer Vision, Real-Time Hand Tracking, Image Processing, Gesture Recognition, and Game Development** into one complete interactive experience.

---

# ✨ Features

## 🎥 Computer Vision Features

✅ Real-time webcam access  
✅ Hand landmark detection using MediaPipe  
✅ Pinch gesture recognition  
✅ Gesture-based image selection  
✅ Automatic image cropping  
✅ Real-time interaction  

## 🧩 Puzzle Game Features

✅ Automatic 3×3 puzzle generation  
✅ Random puzzle shuffling  
✅ Drag-and-drop puzzle solving  
✅ Move counter  
✅ Live timer  
✅ Puzzle preview mode  
✅ Restart functionality  
✅ Puzzle completion detection  
✅ Victory animation 🎉  

---

# 🚀 Project Workflow

```
Launch Application
        ↓
Open Webcam
        ↓
Detect Hand Landmarks
        ↓
Pinch Gesture Detection
        ↓
Select Image Region
        ↓
Capture Selfie
        ↓
Split Image into 9 Pieces
        ↓
Shuffle Puzzle Pieces
        ↓
Launch Puzzle Interface
        ↓
Solve Using Drag & Drop
        ↓
Victory Animation 🎉
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand tracking and gesture recognition |
| NumPy | Image and array manipulation |
| Pygame | Game interface and rendering |

---

# 📂 Project Structure

```
selfie-puzzle-game/

│
├── main.py
├── camera.py
├── gesture.py
├── capture.py
├── puzzle.py
├── game.py
├── requirements.txt
└── README.md
```

---

# 📄 File Description

| File | Responsibility |
|------|---------------|
| main.py | Application entry point and workflow control |
| camera.py | Webcam initialization and frame capture |
| gesture.py | MediaPipe hand tracking and pinch detection |
| capture.py | Selection rectangle and image capture logic |
| puzzle.py | Puzzle generation, shuffling and validation |
| game.py | Pygame interface and puzzle gameplay |

---

# ⚙️ Installation & Setup

## Clone Repository

```bash
git clone https://github.com/mohdkupe07/selfie-puzzle-game.git
```

## Open Project Folder

```bash
cd selfie-puzzle-game
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Git Bash

```bash
source venv/Scripts/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running The Application

Start the application:

```bash
python main.py
```

---

# 🎮 How To Use

## Phase 1 — Capture Selfie

| Gesture | Action |
|---------|--------|
| Pinch & Hold | Start selection rectangle |
| Move while pinching | Resize selection area |
| Release Pinch | Lock selection |
| Pinch inside rectangle | Capture image |

Keyboard Controls:

```
ESC / Q → Exit Application
```

---

# 🧩 Phase 2 — Solve Puzzle

| Action | Result |
|--------|--------|
| Drag Puzzle Piece | Pick and move tile |
| Drop Piece | Swap positions |
| Hold Preview | View original image |
| Restart | Shuffle puzzle again |

---

# ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| camera_index | 0 | Webcam device index |
| pinch_threshold | 0.06 | Gesture sensitivity |
| piece_size | 160 | Puzzle tile size |
| GRID_SIZE | 3 | Puzzle dimension |

---

# 🏗️ System Architecture

```
                         ┌──────────────┐
                         │    main.py   │
                         │ Entry Point  │
                         └──────┬───────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        ┌───────▼────────┐             ┌────────▼────────┐
        │ Capture Phase  │             │ Puzzle Phase    │
        └───────┬────────┘             └────────┬────────┘
                │                               │
     ┌──────────┼──────────┐        ┌───────────┼───────────┐
     │          │          │        │           │           │
┌────▼────┐ ┌───▼──────┐ ┌─▼────┐ ┌─▼────────┐ ┌─▼──────┐ ┌─▼────────┐
│ Camera  │ │ Gesture  │ │Image │ │ Puzzle   │ │Shuffle │ │  Game    │
│ Module  │ │Detector  │ │Capture│ │Generator │ │ Engine │ │Interface │
└─────────┘ └──────────┘ └──────┘ └──────────┘ └────────┘ └──────────┘
```

---

# 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Computer Vision using OpenCV
- Real-Time Hand Gesture Recognition
- Image Processing
- Object-Oriented Programming
- Event Driven Programming
- Human Computer Interaction
- Puzzle Generation Algorithms
- Game Development using Pygame

---

# 🔮 Future Improvements

🚀 Support for 4×4 and 5×5 puzzle modes  
🚀 Online leaderboard system  
🚀 Save best completion times  
🚀 Sound effects and background music  
🚀 Face detection based cropping  
🚀 Mobile application version  
🚀 Multiplayer challenge mode  
🚀 AI-based puzzle difficulty adjustment  

---

# 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Commit changes  
5. Create a Pull Request  

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Mohammed Kupe**

Computer Science Student | AI & Software Development Enthusiast | Computer Vision Developer

⭐ If you like this project, consider giving it a star!
