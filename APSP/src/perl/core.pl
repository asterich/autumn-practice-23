use 5.020;
use utf8;
use warnings;
use autodie;
use feature 'signatures';
use List::Util qw/min/;

my $vertex_num = $ARGV[0];

my @result = ();
for (0 .. $vertex_num - 1) {
    push @result, [];
}

for my $i (0 .. $vertex_num - 1) {
    for my $j (0 .. $vertex_num - 1) {
        my $tmp;
        sysread STDIN, $tmp, 4;
        $result[$i][$j] = unpack 'l', $tmp;
    }
}

for my $k (0 .. $vertex_num - 1) {
    for my $i (0 .. $vertex_num - 1) {
        for my $j (0 .. $vertex_num - 1) {
            $result[$i][$j] = min $result[$i][$j], $result[$i][$k] + $result[$k][$j];
        }
    }
}

for my $i (0 .. $vertex_num - 1) {
    for my $j (0 .. $vertex_num - 1) {
        my $buf = pack 'l', $result[$i][$j];
        syswrite STDOUT, $buf, 4;
    }
}
