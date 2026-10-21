"""Diagnostic-only call boundaries, enabled solely by RELIABILITY_TRACE_DIR."""
import os
import sys
import time

_directory = os.environ.get('RELIABILITY_TRACE_DIR')
if _directory:
    _stream = open(os.path.join(_directory, 'events-%s.tsv' % os.getpid()), 'a', buffering=1)
    _stream.write('%s\tstart\tprocess\t%s\n' % (time.monotonic(), os.getpid()))
    _names = {'load_dataset', 'build_profile', 'fit_predict', 'fit_predict_supervised',
              'fit_predict_routed', 'fit_baseline_supervised', '_fit', '_matrix',
              'write_outputs', 'validate_predictions', '_recover_checkpoint',
              'write_checkpoint', 'load_checkpoint', '_model_worker',
              '_parallel_build_trees', 'summarize_training', 'make_route_plan'}

    def _trace(frame, event, arg):
        if event not in ('call', 'return'):
            return
        code = frame.f_code
        filename = code.co_filename.replace('\\', '/')
        name = code.co_name
        wanted = ('unknown_dataset/' in filename and
                  (name in _names or name in {'fit', 'transform', 'fit_transform'}))
        wanted = wanted or ('sklearn/' in filename and
                            (name == '_parallel_build_trees' or
                             name in {'fit', 'predict'} and any(x in filename for x in
                             ['ensemble/_forest', 'linear_model/', '_hist_gradient_boosting/'])))
        if wanted:
            estimator = frame.f_locals.get('self')
            candidate = frame.f_locals.get('name', '') if name == '_fit' else ''
            _stream.write('%s\t%s\t%s:%s:%s\t%s\n' %
                          (time.monotonic(), event, filename.split('/')[-1], name,
                           type(estimator).__name__ if estimator is not None else candidate,
                           os.getpid()))
    sys.setprofile(_trace)
