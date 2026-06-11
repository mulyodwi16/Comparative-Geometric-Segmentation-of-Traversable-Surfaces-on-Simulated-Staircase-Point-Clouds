# Comparative Geometric Segmentation of Traversable Surfaces on Simulated Staircase Point Clouds

Mini project for the Sinyal dan Sensor course (PENS, Magister Teknik Informatika).
The project generates a synthetic 4-step staircase point cloud and benchmarks four
segmentation methods that distinguish the traversable tread surface from risers
and outliers.

## Methods

1. **RANSAC plane fitting** — constrained iterative plane extraction (horizontal
   then vertical) using normal-orientation filtering.
2. **PCA + normal vector analysis** — per-point local PCA, classification by
   the absolute z-component of the surface normal.
3. **DBSCAN + slope analysis** — DBSCAN in an augmented feature space
   `(x, y, z, w * |n_z|)` followed by per-cluster slope classification.
4. **Height histogram analysis (bonus)** — tread plateau detection from a
   z-axis histogram.

## Evaluation

Per-method metrics (accuracy, precision, recall, F1, IoU) for the traversable
class, plus a noise sweep over LiDAR `sigma in {0, 0.005, 0.01, 0.02, 0.03, 0.05}`.

## Layout

```
MiniProject/
  staircase_segmentation.ipynb   main deliverable notebook
  _dev.py                        development script (mirror of notebook logic)
  _make_notebook.py              notebook builder
  data/                          generated cloud + metrics
  figures/                       rendered figures
  requirements.txt
```

The IEEE conference paper (LaTeX + PDF) is maintained separately, outside this
code repository.

## Reproduce

```
pip install -r requirements.txt
python _dev.py            # or open the notebook
```

## Author

Dwi Mulyo, Department of Informatics and Computer Engineering, Politeknik
Elektronika Negeri Surabaya. `mulyodwi16@pasca.student.pens.ac.id`
