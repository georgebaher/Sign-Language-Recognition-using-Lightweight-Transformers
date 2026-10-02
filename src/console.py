"""Terminal output for FG2027: colour and a live progress line on a terminal, plain lines in logs."""
import os
import shutil
import sys

LIVE = sys.stdout.isatty()
COLOR = LIVE and 'NO_COLOR' not in os.environ
WIDTH = min(shutil.get_terminal_size((88, 24)).columns, 88)


def paint(text, code):
    return f'\033[{code}m{text}\033[0m' if COLOR else str(text)


def bold(text): return paint(text, '1')
def dim(text): return paint(text, '2')
def green(text): return paint(text, '32')
def yellow(text): return paint(text, '33')
def red(text): return paint(text, '31')
def cyan(text): return paint(text, '36')


def line(text=''):
    """Print a permanent line, first clearing any live progress line."""
    if LIVE:
        sys.stdout.write('\r\033[K')
    print(text, flush=True)


def live(text):
    """Overwrite the single live progress line (terminal only)."""
    if LIVE:
        sys.stdout.write('\r\033[K' + text)
        sys.stdout.flush()


def rule(title):
    line(bold(cyan(f'━━ {title} ' + '━' * max(WIDTH - len(title) - 4, 2))))


def duration(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds}s'
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f'{minutes}m{seconds:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m'


def bar(done, total, width=20):
    filled = round(width * done / total)
    return '▕' + '█' * filled + '░' * (width - filled) + '▏'


def dataset_summary(dataset, report):
    rule(dataset)
    counts, missing = report['counts'], report['missingness']
    gaps = ' · '.join(f"{name} {missing[f'{key}_landmarks']['missing_coordinate_fraction']:.1%}"
                      for name, key in (('hands', 'hand'), ('pose', 'pose'), ('face', 'face')))
    line(f"  {dim('clips')}     {counts['train']} train · {counts['val']} val · {counts['test']} test"
         f"    {dim('frames')} {report['length_min']}–{report['length_max']} per clip")
    line(f"  {dim('missing')}   {gaps}    {dim('multi-person frames')} {report['person_selection']['frames']}")
    line()


EPOCH_HEADER = (f"  {'epoch':>7}   {'train loss':>10} {'acc':>6} │ {'val loss':>8} {'acc':>6} │"
                f" {'best':>6} {'':<5} {'time':>6} {'eta':>6}")


def epoch_row(row, total, best_accuracy, best_epoch, elapsed):
    eta = elapsed / row['epoch'] * (total - row['epoch'])
    return (f"  {row['epoch']:>3}/{total:<3}   {row['train_loss']:>10.3f} {row['train_acc']:>6.1%} │"
            f" {row['val_loss']:>8.3f} {row['val_acc']:>6.1%} │ {best_accuracy:>6.1%} {'@' + str(best_epoch):<5}"
            f" {duration(elapsed):>6} {duration(eta):>6}")


def epoch_live(row, total, best_accuracy, best_epoch, elapsed):
    eta = elapsed / row['epoch'] * (total - row['epoch'])
    return (f"  {cyan(bar(row['epoch'], total))} {row['epoch']:>3}/{total}  "
            f"train {row['train_acc']:.1%} │ val {row['val_acc']:.1%} │ best {best_accuracy:.1%} @{best_epoch}"
            f"  {dim('eta ' + duration(eta))}")


def table(header, rows):
    """Align rows of strings under a header; 'Pending' cells are dimmed."""
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *rows)]
    line('  ' + '  '.join(bold(str(cell).ljust(width)) for cell, width in zip(header, widths)))
    for row in rows:
        line('  ' + '  '.join((dim if cell == 'Pending' else str)(str(cell).ljust(width))
                              for cell, width in zip(row, widths)))
