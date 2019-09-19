""" This module contains operations regarding training configurations.
"""

import tensorflow as tf
import modules.grid
import os
import json
from typing import Optional
from datetime import datetime


class Config:

    def __init__(
            self,
            base_dir: str,
            grid_id: str,
            encoder_id: str,
            decoder_id: str,
            filler_id: Optional[str],
            loss_encoder: str,
            loss_decoder: str,
            learning_rate_encoder: float,
            learning_rate_decoder: float,
            batch_size: int = 500,
            num_epochs: int = 1,
            config_id: Optional[str]=None
    ):
        self.base_dir: str = base_dir
        self.grid_id: str = grid_id
        self.encoder_id: str = encoder_id
        self.decoder_id: str = decoder_id
        self.filler_id: Optional[str] = filler_id
        self.loss_encoder: str = loss_encoder
        self.loss_decoder: str = loss_decoder
        self.learning_rate_encoder: float = learning_rate_encoder
        self.learning_rate_decoder: float = learning_rate_decoder
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs

        self.config_id = config_id if config_id is not None else 'C-' + datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S-%f')

        self.grid: modules.grid.AbstractGrid = self._load_grid()
        self.encoder: tf.keras.Model = self._load_encoder()
        self.decoder: tf.keras.Model = self._load_decoder()
        self.filler = self._load_filler()

    @classmethod
    def load(cls, base_dir: str, config_id: str):
        """ Loads a configuration given a base directory of the save file and the config id.
        """
        with open(os.path.join(base_dir, 'output', config_id, config_id + '.json')) as infile:
            data_dict = json.load(infile)
        print(data_dict)
        return cls(**data_dict)

    def _load_grid(self) -> modules.grid.AbstractGrid:
        """ Loads a grid from grids/ folder in base directory.
        """
        return modules.grid.load(
            os.path.join(
                self.base_dir,
                'grids',
                self.grid_id + '.pkl'
            )
        )

    def _load_encoder(self) -> tf.keras.Model:
        """ Loads an encoder from models/encoders/ folder in base directory.
        """
        with open(
            os.path.join(
                self.base_dir,
                'models',
                'json',
                'encoders',
                self.encoder_id + '.json'
            )
        ) as infile:
            contents = infile.read()
            encoder = tf.keras.models.model_from_json(contents)
        return encoder

    def _load_decoder(self) -> tf.keras.Model:
        """ Loads a decoder from models/decoders/ folder in base_directory.
        """
        with open(
            os.path.join(
                self.base_dir,
                'models',
                'json',
                'decoders',
                self.decoder_id + '.json'
            )
        ) as infile:
            contents = infile.read()
            decoder = tf.keras.models.model_from_json(contents)
        return decoder

    def _load_filler(self) -> tf.keras.Model:
        """ Loads a filler from models/decoders/ folder in base_directory
        """
        if self.filler_id is None:
            return lambda renders: renders
        else:
            return tf.keras.models.load_model(
                os.path.join(
                    self.base_dir,
                    'models',
                    'trained',
                    'fillers',
                    self.filler_id + '.h5'
                )
            )

    def save(self) -> None:
        save_dir = os.path.join(self.base_dir, 'output', self.config_id)
        os.mkdir(save_dir)
        save_file_path = os.path.join(save_dir, self.config_id + '.json')

        data_dict = {
            'base_dir': self.base_dir,
            'config_id': self.config_id,
            'grid_id': self.grid_id,
            'encoder_id': self.encoder_id,
            'decoder_id': self.decoder_id,
            'filler_id': self.filler_id,
            'loss_encoder': self.loss_encoder,
            'loss_decoder': self.loss_decoder,
            'learning_rate_encoder': self.learning_rate_encoder,
            'learning_rate_decoder': self.learning_rate_decoder,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
        }

        with open(save_file_path, 'w+') as outfile:
            json.dump(data_dict, outfile, sort_keys=True, indent=4)
