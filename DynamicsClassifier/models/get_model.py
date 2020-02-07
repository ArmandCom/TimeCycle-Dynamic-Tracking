from .DynClass import DynClass

def get_model(opt):

  model = DynClass(opt)
  model.setup_training()
  model.initialize_weights()
  return model
