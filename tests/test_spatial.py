import pytest
from cv_pipeline.spatial import SpatialMapper

def test_spatial_mapper_5_zones():
    mapper = SpatialMapper(frame_width=1000, n_zones=5)
    # Zone width = 200
    # Zones: 0-199 (Left), 200-399 (C-L), 400-599 (Center), 600-799 (C-R), 800-999 (Right)
    
    # Test Left
    assert mapper.get_zone((0, 0, 100, 100)) == "Left" # CenterX = 50
    # Test Center-Left
    assert mapper.get_zone((250, 0, 350, 100)) == "Center-Left" # CenterX = 300
    # Test Center
    assert mapper.get_zone((450, 0, 550, 100)) == "Center" # CenterX = 500
    # Test Right
    assert mapper.get_zone((900, 0, 1000, 100)) == "Right" # CenterX = 950
    
def test_spatial_mapper_out_of_bounds():
    mapper = SpatialMapper(frame_width=1000, n_zones=5)
    # Clamp test
    assert mapper.get_zone((1000, 0, 1100, 100)) == "Right" # CenterX = 1050 (should clamp)
    assert mapper.get_zone((-100, 0, 0, 100)) == "Left" # CenterX = -50 (should clamp)
