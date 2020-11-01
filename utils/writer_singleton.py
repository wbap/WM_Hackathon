from torch.utils.tensorboard import SummaryWriter


class WriterSingleton:
  writer = None
  global_step = 0

  def __init__(self):
    pass

  @staticmethod
  def get_writer():
    if not WriterSingleton.writer:
      WriterSingleton.writer = SummaryWriter()
      print("---------------> Created writer at logdir(): ", WriterSingleton.writer.get_logdir())
    return WriterSingleton.writer
