import torch
import yaml
#from unittest import Assert

from postprocessing.visualization import get_hp_z_position, get_hp_z_position_from_info, remove_z_dimension


def test_get_hp_z_position():
    #Fixture
    tensor=torch.zeros((5, 5, 5))
    tensor[1,2,3] = 1

    # Expected result
    expected = 3
    # Actual result
    actual=get_hp_z_position(tensor)
    # Test
    assert actual==expected, "Result is not the expected z position"

def test_get_hp_z_position_from_info():
    #Fixture
    with open("unittests/dummyInfo.yaml", "r") as f:
            info = yaml.safe_load(f)
    tensor = torch.Tensor([[
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]],
        [[1,0,0], [0,0,0], [0,0,0]]],
        [
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,1,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]]],
        [#sdf
        [[0,0,1], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]]],
        [#Material ID
        [[0,0,1], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]]]])
    
    # Expected result
    expected = 2
    # Actual result
    actual=get_hp_z_position_from_info(tensor,info)
    # Test
    assert actual==expected, "Result is not the expected z position"

def test_remove_z_dimension_3d():
    #Fixture
    #4x2x3
    tensor = torch.Tensor([
        [[0,0,1], [0,0,1]],
        [[0,0,1], [0,0,1]],
        [[0,0,1], [0,0,1]],
        [[0,0,1], [0,0,1]]])
    
    z_position=2
    
    # Expected result
    expected = torch.ones((4,2))
    # Actual result
    actual=remove_z_dimension(tensor,z_position)
    # Test
    assert torch.allclose(actual, expected), f"z dim could not be removed at z position {z_position}"

def test_remove_z_dimension_4d():
     #Fixture
    tensor=torch.zeros((4, 2, 3, 4))
    tensor[:,:,:,0]=5
    tensor[:,:,:,1]=6
    tensor[:,:,:,2]=7
    tensor[:,:,:,3]=8

    z_position=0
    
    # Expected result
    expected = tensor[:,:,:,0]
    # Actual result
    actual=remove_z_dimension(tensor,z_position)
    # Test
    assert torch.allclose(actual, expected), f"z dim could not be removed at z position {z_position}"

if __name__ == "__main__":
    test_get_hp_z_position()
    test_get_hp_z_position_from_info()
    test_remove_z_dimension_3d()
    test_remove_z_dimension_4d()