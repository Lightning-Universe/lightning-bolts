import re
from warnings import warn

from pytorch_lightning.utilities.exceptions import MisconfigurationException


class YoloConfiguration:
    def __init__(self, path: str):
        """
        Parser for YOLOv4 network configuration files.

        Saves the variables from the first configuration section to attributes of this object, and
        the rest of the sections to the `modules` list.

        Args:
            path (str): configuration file to read
        """
        with open(path, 'r') as config_file:
            sections = self._read_file(config_file)

        if len(sections) < 2:
            raise MisconfigurationException(
                "The model configuration file should include at least two sections.")

        self.__dict__.update(sections[0])
        self.modules = sections[1:]

    def _read_file(self, config_file):
        """
        Reads a YOLOv4 network configuration file and returns a list of configuration sections.

        Args:
            config_file (iterable over lines): The configuration file to read.

        Returns:
            sections (list): A list of configuration sections.
        """
        section_re = re.compile(r'\[([^]]+)\]')
        list_variables = ('layers', 'anchors', 'mask', 'scales')
        variable_types = {
            'activation': str,
            'anchors': int,
            'angle': float,
            'batch': int,
            'batch_normalize': bool,
            'beta_nms': float,
            'burn_in': int,
            'channels': int,
            'classes': int,
            'cls_normalizer': float,
            'decay': float,
            'exposure': float,
            'filters': int,
            'from': int,
            'groups': int,
            'group_id': int,
            'height': int,
            'hue': float,
            'ignore_thresh': float,
            'iou_loss': str,
            'iou_normalizer': float,
            'iou_thresh': float,
            'jitter': float,
            'layers': int,
            'learning_rate': float,
            'mask': int,
            'max_batches': int,
            'max_delta': float,
            'momentum': float,
            'mosaic': bool,
            'nms_kind': str,
            'num': int,
            'obj_normalizer': float,
            'pad': bool,
            'policy': str,
            'random': bool,
            'resize': float,
            'saturation': float,
            'scales': float,
            'scale_x_y': float,
            'size': int,
            'steps': str,
            'stride': int,
            'subdivisions': int,
            'truth_thresh': float,
            'width': int
        }

        section = None
        sections = []

        def convert(key, value):
            """Converts a value to the correct type based on key."""
            if not key in variable_types:
                warn('Unknown YOLO configuration variable: ' + key)
                return key, value
            if key in list_variables:
                value = [variable_types[key](v) for v in value.split(',')]
            else:
                value = variable_types[key](value)
            return key, value

        for line in config_file:
            line = line.strip()
            if (not line) or (line[0] == '#'):
                continue

            section_match = section_re.match(line)
            if section_match:
                if section is not None:
                    sections.append(section)
                section = {'type': section_match.group(1)}
            else:
                key, value = line.split('=')
                key = key.rstrip()
                value = value.lstrip()
                key, value = convert(key, value)
                section[key] = value
        if section is not None:
            sections.append(section)

        return sections
