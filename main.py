"""
Algorithm server definition.
Documentation: https://github.com/Imaging-Server-Kit/cookiecutter-serverkit
"""
from typing import List, Literal, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import skimage.io
import imaging_server_kit as serverkit

# import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_napari_tracks

class Parameters(BaseModel):
    """Defines the algorithm parameters"""
    image: str = Field(
        title="Image",
        description="Input image (2D, 3D).",
        json_schema_extra={"widget_type": "image"},
    )
    mask: str = Field(
        title="Mask",
        description="Input segmentation mask (2D, 3D).",
        json_schema_extra={"widget_type": "mask"},
    )
    mode: Literal['greedy', 'greedy_nodiv'] = Field(
        default='greedy',
        title="Mode",
        description="Tracking mode.",
        json_schema_extra={"widget_type": "dropdown"},
    )

    @field_validator("image", mode="after")
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim not in [2, 3]:
            raise ValueError("Array has the wrong dimensionality.")
        return image_array

    @field_validator("mask", mode="after")
    def decode_mask_array(cls, v) -> np.ndarray:
        mask_array = serverkit.decode_contents(v)
        if mask_array.ndim not in [2, 3]:
            raise ValueError("Array has the wrong dimensionality.")
        return mask_array

class Server(serverkit.Server):
    def __init__(
        self,
        algorithm_name: str="trackastra",
        parameters_model: Type[BaseModel]=Parameters
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        mode: str,
        **kwargs
    ) -> List[tuple]:
        """Runs the algorithm."""

        device = "cpu"  # For now - To avoid CUDA errors

        model = Trackastra.from_pretrained("general_2d", device=device)

        track_graph = model.track(image, mask, mode=mode)  # or mode="ilp", or "greedy_nodiv"

        napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)
        
        return [(napari_tracks, {"name": "Tracks"}, "tracks")]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Loads one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images

server = Server()
app = server.app

if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)