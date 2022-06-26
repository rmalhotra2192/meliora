=======
About the package
=======

**credit-tests** is a Python package that provides a set of statistical tests and tools to assess the performance of the credit risk models. The aim of the package is to provide all common tests used by today's modellers when developing, maintaining and validating their PD, LGD, EAD and prepayment models. All tests have been thoroughly tested and documented. Whenever possible, the definition of the test was retrieved from the authoritive source like EBA, ECB or Basel Committee.

The main contributors started building their first statistical credit models back in 2003. Over the years, we have impemented similar set of test in several different financial institutions. 

The source code has been developed specifically for this package from scratch.

Main aim
-----------------
The aim of this package is to provide credit risk practioners with the tools to develop their credit risk models without the need to implement standard tooling. All tests should be covered with unit tests and algorithms should be replicated using other tools to avoid errors.

Main Features
-----------------
  - tests cover both IFRS 9 and IRB models as well as non-regulatory models
  - the tool contains more than 30 tests
  - all test have been covered with unit tests 
  - the tests have been documented in detail
  - commonly accepted tresholds have been provided for convenience purposes

  For the list of all tests, see Overview > List of tests
  

Dependencies
------------------------
- NumPy (https://www.numpy.org)
- pandas (https://www.numpy.org)
- statsmodels (https://www.numpy.org)
- Scikit-learn (https://www.numpy.org)


Getting Help
------------------

For usage questions, send an email to anton.treialt@aistat.com

License
----------------------
MIT LIcense