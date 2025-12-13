for i  in {1..10}; do
	echo  "-- running --"
	make test_all
	echo "-- finish iteration -- "
done
