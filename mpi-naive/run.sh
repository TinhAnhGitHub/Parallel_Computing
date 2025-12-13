for i  in {1..5}; do
	echo  "-- running --"
	make test_all
	echo "-- finish iteration -- "
done
