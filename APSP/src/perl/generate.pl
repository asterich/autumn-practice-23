#!/usr/bin/env perl
use 5.020;
use utf8;
use warnings;
use autodie;
use feature 'signatures';
use open qw(:std :utf8);

open STDOUT, '>', 'apsp.cc';

print 'const char *perl_program = "';

open my $perl, '<', 'core.pl';
while (<$perl>) {
    chomp;
    s/\\/\\\\/g;
    s/"/\\"/g;
    s/$/\\n/;
    print;
}

print '";';

say '';

open my $wrapper, '<', 'wrapper.cc';
print while <$wrapper>;
