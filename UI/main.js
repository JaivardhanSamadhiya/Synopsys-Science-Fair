        function scrollToSection(sectionId) {
            var section = document.getElementById(sectionId);
            if (section) {
                section.scrollIntoView({ behavior: 'smooth' });
            }
        }

        function openAIMODEL() {
            window.open('AI-Model.html');
            return false;
        }

        function openContact() {
            window.open('contact.html');
            return false;
        }

            // Hover effects
            document.getElementById(id).addEventListener('mouseover', function() {
                document.getElementById(`${id}-btn`).style.display = 'block';
            });
            document.getElementById(id).addEventListener('mouseout', function() {
                document.getElementById(`${id}-btn`).style.display = 'none';
            });
            document.getElementById(`${id}-btn`).addEventListener('mouseover', function() {
                this.style.display = 'block';
            });

        // Scroll to top button
        window.addEventListener("scroll", function() {
            var scrollToTopBtn = document.getElementById("scrollToTopBtn");
            if (window.scrollY > 50) { 
                scrollToTopBtn.style.display = "block";
            } else {
                scrollToTopBtn.style.display = "none";
            }
        });
        
        document.getElementById("scrollToTopBtn").addEventListener("click", function() {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });        