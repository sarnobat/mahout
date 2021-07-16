{
	if (NF == 3) {
		if ($(2) > 20) {
			print $2,$1,$3 
		}
	}
	else { 
		print "Number of fields: " NF ": " $0| "cat 1>&2" 
	}
}