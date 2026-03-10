from typing import Dict, Any, Optional
from datasets import load_dataset, DatasetDict, Audio

class DatasetFactory:
    @staticmethod
    def get_dataset(
        dataset_type: str, 
        path: str, 
        name: Optional[str] = None, 
        split: Optional[str] = None,
        sampling_rate: int = 16000,
        **kwargs
    ) -> Any:
        """
        Factory to load datasets from multiple sources.
        
        Args:
            dataset_type: 'huggingface' (HuggingFace Hub), 'common_voice', or 'custom' (CSV/TSV/JSON)
            path: Repository ID or local path to files
            name: Specific configuration name for HF/CV datasets
            split: Training/Validation/Test split
            sampling_rate: Audio sampling rate (Whisper requires 16k)
        """
        if dataset_type in ['huggingface', 'common_voice']:
            # Load from Hub
            dataset = load_dataset(path, name=name, split=split, **kwargs)
        elif dataset_type == 'custom':
            # Load from local files (CSV/TSV/JSON)
            extension = path.split('.')[-1]
            if extension == 'tsv':
                dataset = load_dataset('csv', data_files=path, delimiter='\t', split=split, **kwargs)
            else:
                dataset = load_dataset(extension, data_files=path, split=split, **kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Cast audio column to the correct sampling rate
        # Assuming the audio column name is 'audio' or 'path'
        audio_column = kwargs.get("audio_column", "audio")
        if audio_column in dataset.column_names:
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
        
        return dataset

def load_data_from_config(config: Dict[str, Any]) -> DatasetDict:
    """Convenience function to load training, validation, and test sets."""
    datasets = {}
    for split in ['train', 'validation', 'test']:
        if split in config:
            datasets[split] = DatasetFactory.get_dataset(
                split=split,
                **config[split]
            )
    return DatasetDict(datasets)
