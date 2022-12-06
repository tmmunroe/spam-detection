from setuptools import setup

setup(name='sms_spam_classifier_utils',
      version='0.1',
      description='Helpers used to train SMS Spam classifier',
      url='http://github.com/tmmunroe/sms_spam_classifier',
      author='Turner Mandeville',
      author_email='t.m.munroe@gmail.com',
      license='MIT',
      packages=['sms_spam_classifier_utils'],
      install_requires=[
        'numpy'
      ],
      zip_safe=False)