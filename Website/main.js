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

        function openFeatures() {
            window.open('features.html');
        }

        function openRosacea() {
            window.open('contact.html');
            return false;
        }

        // Buy button event listeners
        ['sigma', 'ohio', 'fish', 'fanum'].forEach(id => {
            document.getElementById(`${id}-btn`).addEventListener('click', function() {
                Swal.fire({
                    position: 'top',
                    icon: 'success',
                    title: 'Purchase Successful',
                    text: 'One item added to cart.',
                    showConfirmButton: false,
                    timer: 2000
                });
            });

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
        });

        // Scroll to top button
        window.addEventListener("scroll", function() {
            var scrollToTopBtn = document.getElementById("scrollToTopBtn");
            if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
                scrollToTopBtn.style.display = "block";
            } else {
                scrollToTopBtn.style.display = "none";
            }
        });

        document.getElementById("scrollToTopBtn").addEventListener("click", function() {
            document.body.scrollTop = 0;
            document.documentElement.scrollTop = 0;
        });

        // Submit button display
        document.getElementById('submit').addEventListener('mouseover', function() {
            this.style.display = 'block';
        });
        document.getElementById('submit').addEventListener('mouseout', function() {
            this.style.display = 'block';
        });