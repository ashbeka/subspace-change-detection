@echo off

python visual_contribution_first.py ../dataset/16_57.c3d
python visual_contribution_second_ds.py ../dataset/16_57.c3d
python visual_contribution_geodesic.py ../dataset/16_57.c3d

python visual_contribution_first.py ../dataset/02_01.c3d
python visual_contribution_second_ds.py ../dataset/02_01.c3d
python visual_contribution_geodesic.py ../dataset/02_01.c3d

python visual_contribution_first.py ../dataset/07_01.c3d
python visual_contribution_second_ds.py ../dataset/07_01.c3d
python visual_contribution_geodesic.py ../dataset/07_01.c3d
