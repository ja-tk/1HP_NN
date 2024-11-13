import torch
import yaml

from data_stuff.utils_3d import get_hp_position, get_hp_position_from_input, remove_first_dim
#to start, move file into parent folder

def test_get_hp_position():
    #Fixture
    tensor=torch.zeros((5, 5, 5))
    tensor[3,2,1] = 1

    # Expected result
    expected = 3
    # Actual result
    actual=get_hp_position(tensor)[0]
    # Test
    assert actual==expected, "Result is not the expected position"

def test_get_hp_position_from_input():
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
        [#sdf-tensor-dummy: max at hp_position (0,1,2)
        [[0,0,0], [0,0,1], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]]],
        [#Material ID-tensor-dummy: max at hp_position (0,1,2)
        [[0,0,0], [0,0,1], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0], [0,0,0]]]])
    
    # Expected result
    expected = 0
    # Actual result
    actual=get_hp_position_from_input(tensor,info)[0]
    # Test
    assert actual==expected, "Result is not the expected position"

def test_remove_first_dimension_3d():
    #Fixture
    #4x2x3
    tensor = torch.Tensor([
        [[0,0,0], [0,0,0]],
        [[0,0,0], [0,0,0]],
        [[1,1,1], [1,1,1]],
        [[0,0,0], [0,0,0]]])
    
    position=2

    # Expected result
    expected = torch.ones((2,3))
    # Actual result
    actual=remove_first_dim(tensor,position)
    # Test
    assert torch.allclose(actual, expected), f"z dim could not be removed at position {position}"

def test_remove_dimension_4d():
     #Fixture
    tensor=torch.zeros((4, 2, 3, 4))
    tensor[:,0,:,:]=5
    tensor[:,1,:,:]=6

    position=0
    
    # Expected result
    expected = tensor[:,0,:,:]
    # Actual result
    actual=remove_first_dim(tensor,position)
    # Test
    assert torch.allclose(actual, expected), f"z dim could not be removed at position {position}"


if __name__ == "__main__":
    print("tests start")
    test_get_hp_position()
    test_get_hp_position_from_input()
    test_remove_first_dimension_3d()
    test_remove_dimension_4d()
    print("tests finished")