import argparse
import os
import shutil
import yaml


def copy(src, dest, directory, required):
  ds = 'Directory' if directory else 'File'
  def handle_error(e):
    if required:
      print('%s not copied. Error: %s' % (ds, e))
      print('Exiting, SUBMISSION NOT ZIPPED!')
      shutil.rmtree('temp_submission', ignore_errors=True)
      exit()
    else:
      print('%s %s missing but optional, skipping.' % (ds, src))
  try:
    if directory:
      shutil.copytree(src, dest)
    else:
      shutil.copy(src, dest)
  except shutil.Error as e:
    handle_error(e)
  except OSError as e:
    handle_error(e)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--gt_username', required=True, type=str)

  args = parser.parse_args()

  shutil.rmtree('temp_submission', ignore_errors=True)
  os.mkdir('temp_submission')

  dir_list = yaml.load(open('proj2_configs/zip_dir_list.yml'), Loader=yaml.BaseLoader)
  for file_name in dir_list['required_files']:
    shutil.copy(file_name, './temp_submission')

  out_file = args.gt_username + '.zip'
  shutil.make_archive(args.gt_username, 'zip', 'temp_submission')
  if(os.path.getsize(out_file) > 50000000):
    os.remove(out_file)
    print("SUBMISSION DID NOT ZIP, ZIPPED SIZE > 50MB")
  shutil.rmtree('temp_submission', ignore_errors=True)
