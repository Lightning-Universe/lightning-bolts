import copy
from itertools import zip_longest
from typing import List, Any, Dict, Callable

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info


class PrintTableMetricsCallback(Callback):
    def __init__(self):
        """
        Prints a table with the metrics in columns on every epoch end

        Example::

            from pl_bolts.callbacks import PrintTableMetricsCallback

            callback = PrintTableMetricsCallback()

        pass into trainer like so:

        .. code-block:: python

            trainer = pl.Trainer(callbacks=[callback])
            trainer.fit(...)

            # ------------------------------
            # at the end of every epoch it will print
            # ------------------------------

            # loss│train_loss│val_loss│epoch
            # ──────────────────────────────
            # 2.2541470527648926│2.2541470527648926│2.2158432006835938│0

        """
        self.metrics = []

    def on_epoch_end(self, trainer, pl_module):
        metrics_dict = copy.copy(trainer.callback_metrics)
        self.metrics.append(metrics_dict)
        rank_zero_info(dicts_to_table(self.metrics))


def dicts_to_table(dicts: List[Dict],
                   keys: List[str] = None,
                   pads: List[str] = None,
                   fcodes: List[str] = None,
                   convert_headers: Dict[str, Callable] = None,
                   header_names: List[str] = None,
                   skip_none_lines: bool = False,
                   replace_values: Dict[str, Any] = None):
    """
    Generate ascii table from dictionary
    Taken from (https://stackoverflow.com/questions/40056747/print-a-list-of-dictionaries-in-table-form)

    Args:
        dicts: input dictionary list; empty lists make keys OR header_names mandatory
        keys: order list of keys to generate columns for; no key/dict-key should
            suffix with '____' else adjust code-suffix
        pads: indicate padding direction and size, eg <10 to right pad alias left-align
        fcodes: formating codes for respective column type, eg .3f
        convert_headers: apply converters(dict) on column keys k, eg timestamps
        header_names: supply for custom column headers instead of keys
        skip_none_lines: skip line if contains None
        replace_values: specify per column keys k a map from seen value to new value;
                        new value must comply with the columns fcode; CAUTION: modifies input (due speed)
    """
    # optional arg prelude
    if keys is None:
        if len(dicts) > 0:
            keys = dicts[0].keys()
        elif header_names is not None:
            keys = header_names
        else:
            raise ValueError('keys or header_names mandatory on empty input list')
    if pads is None:
        pads = [''] * len(keys)
    elif len(pads) != len(keys):
        raise ValueError(f'bad pad length {len(pads)}, expected: {len(keys)}')
    if fcodes is None:
        fcodes = [''] * len(keys)
    elif len(fcodes) != len(fcodes):
        raise ValueError(f'bad fcodes length {len(fcodes)}, expected: {len(keys)}')
    if convert_headers is None:
        convert_headers = {}
    if header_names is None:
        header_names = keys
    if replace_values is None:
        replace_values = {}
    # build header
    headline = '│'.join(f"{v:{pad}}" for v, pad in zip_longest(header_names, pads))
    underline = '─' * len(headline)
    # suffix special keys to apply converters to later on
    marked_keys = [h + '____' if h in convert_headers else h for h in keys]
    marked_values = {}
    s = '│'.join(f"{{{h}:{pad}{fcode}}}" for h, pad, fcode in zip_longest(marked_keys, pads, fcodes))
    lines = [headline, underline, ]
    for d in dicts:
        none_keys = [k for k, v in d.items() if v is None]
        if skip_none_lines and none_keys:
            continue
        elif replace_values:
            for k in d.keys():
                if k in replace_values and d[k] in replace_values[k]:
                    d[k] = replace_values[k][d[k]]
                if d[k] is None:
                    raise ValueError(f"bad or no mapping for key '{k}' is None. Use skip or change replace mapping.")
        elif none_keys:
            raise ValueError(f'keys {none_keys} are None in {d}. Do skip or use replace mapping.')
        for h in convert_headers:
            if h in keys:
                converter = convert_headers[h]
                marked_values[h + '____'] = converter(d)
        line = s.format(**d, **marked_values)
        lines.append(line)
    return '\n'.join(lines)


if __name__ == '__main__':
    a = {'a': 1, 'b': 2}
    b = {'a': 3, 'b': 4}

    print(dicts_to_table([a, b]))
