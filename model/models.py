
def create_model(opt):
    print(opt.model)
    if opt.model == 'full':
        from .full_model import FullModel
        model = FullModel()

    elif opt.model == 'cl_dec':
        from .cl_dec_model import CLDecModel
        model = CLDecModel()

    elif opt.model == 'full_fine':
        from .full_fine_model import FullFineModel
        model = FullFineModel()

    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model