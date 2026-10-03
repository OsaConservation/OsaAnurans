const species_path = "specieslist.csv"


function parseFile(){
    // Fetch and parse the CSV file
        Papa.parse(species_path, {
            delimiter: "\t",
            download: true,
            header: true, // Uses the first row as column headers
            complete: function(results) {
                // const data = results.data.sort((a,b) => a.scientific_name.localeCompare(b.scientific_name));
                const data = []
                const all_names = []

                // Loop through each row and extract the 'scientific_name' column
                results.data.forEach(row => {
                    if(row.display === '1'){
                        data.add(row)
                        all_names.add(row.scientific_name)
                    }    
                });

                let index = Math.floor(Math.random() * all_names.length)

                path = generateQuestion(index, data)

                // filter data where display ==1
                // randomly select species
                // randomly select image/audio 
                // get species name + three random species for answers
                // display all 
                // function to check if clicked answer is correct
                //     Display response / frog picture?
                // generate new question 
                //     Track # correct?
            }
        })
}

function generateQuestion(index, data){

}

function generateAnswers(scientific_name, all_names){

}

function checkAnswer(answer){

}