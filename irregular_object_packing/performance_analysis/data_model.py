# %%
from dataclasses import dataclass, fields
from typing import List

from pandas import DataFrame, read_csv

VERSION = "0.1.2"


@dataclass
class DataModel:
    def __init__(self, dict: dict) -> None:
        for key in dict:
            setattr(self, key, dict[key])
        pass



    def input_headers() -> List[str]:
        return [field.name for field in fields(DataModel) if field.compare]

    @staticmethod
    def from_dict(data: dict):
        """Create a data model from a dict."""
        return DataModel(dict=data)

    @staticmethod
    def data_from_csv(file_path: str) -> list:
        """Return a list of data models from a csv file using pandas."""
        df = read_csv(file_path)
        return [DataModel.from_dict(entry) for entry in df.to_dict(orient="records")]

    def to_df(self):
        return DataFrame([self])

    def save_to_csv(self, file_path: str):
        """Save the data model to a csv file."""
        # check if the file exists
        self.to_df().to_csv(file_path, mode='a', header=False, index=False)

    def set_results(
            self, run_time: float = 0.0, part_time: float = 0.0, scatter_time: float = 0.0, bandwidth: float = 0.0,
            node_unactive_time: float = 0.0, message_count: float = 0.0,
            resource_utilisation: float = 0.0):

        self.run_time = run_time
        self.scatter_time = scatter_time
        self.part_time = part_time
        self.bandwidth = bandwidth
        self.node_unactive_time = node_unactive_time
        self.message_count = message_count
        self.resource_utilisation = resource_utilisation


# %%
