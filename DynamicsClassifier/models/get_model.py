from .DynClass import DynClass

def get_model(opt):
  # if opt.model == 'crop':
  #   model = LDE(opt)
  # else:
  #   raise NotImplementedError

  model = DynClass(opt)
  model.setup_training()
  model.initialize_weights()
  return model
