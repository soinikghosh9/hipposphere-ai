
import React from 'react';
import { Section } from '../components/common/Section';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { TeamMember } from '../types';
import { getMockTeamMembers } from '../services/geminiService';
import { PICSUM_SEEDS, PLACEHOLDER_IMAGE_DIMENSIONS } from '../constants';
import { UsersIcon, LeafIcon, ExternalLinkIcon, PlusCircleIcon } from '../components/common/Icons';

const TeamMemberCard: React.FC<{ member: TeamMember }> = ({ member }) => (
  <Card className="text-center bg-earthy-brown/40">
    <img
      src={member.imageUrl} // This will now come from getMockTeamMembers with local paths
      alt={member.name}
      className="w-32 h-32 object-cover rounded-full mx-auto mb-4 shadow-lg"
    />
    <h4 className="text-xl font-semibold text-sky-blue">{member.name}</h4>
    <p className="text-accent-teal mb-2">{member.role}</p>
    <p className="text-gray-300 text-sm">{member.bio}</p>
  </Card>
);

export const CommunityScreen: React.FC = () => {
  const teamMembers = getMockTeamMembers();

  // USER INSTRUCTION: Create 'community-support-bg.png' in your './public/images/' folder.
  return (
    <div>
      <Section
        title="The Community Spring"
        subtitle="Connect with the HippoSphere AI project, support our hippos, and learn how you can contribute to conservation."
      >
        <Card
            className="mb-8 bg-jungle-green/40 text-center bg-cover bg-center"
            style={{ backgroundImage: `linear-gradient(rgba(16, 79, 85, 0.7), rgba(89, 69, 69, 0.7)), url(./public/images/community-support-bg.png)` }} // Updated path
        >
          <div className="relative z-10 p-6">

            <PlusCircleIcon className="w-16 h-16 text-sunrise-orange mx-auto mb-4" />
            <h3 className="text-2xl font-semibold text-sky-blue mb-3">Support Moodeng & Piko</h3>
            <p className="text-gray-200 mb-6 max-w-xl mx-auto">
              Your contributions help provide Moodeng and Piko with the best possible care, including nutritious food, enriching habitats, and veterinary support. Donations also fund vital conservation programs for pygmy hippos in the wild.
            </p>
            <Button
              variant="primary"
              size="lg"
              onClick={() => alert("Redirecting to donation page (conceptual)...")}
            >
              Donate Now
            </Button>
          </div>
        </Card>

        <Section title="Meet the Team" titleClassName="text-2xl font-semibold text-accent-teal mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {teamMembers.map(member => <TeamMemberCard key={member.id} member={member} />)}
          </div>
        </Section>

        <Section title="Learn More & Get Involved" titleClassName="text-2xl font-semibold text-accent-teal mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-earthy-brown/40">
              <UsersIcon className="w-10 h-10 text-accent-teal mb-3" />
              <h4 className="text-xl font-semibold text-sky-blue mb-2">Visit the Zoo/Sanctuary</h4>
              <p className="text-gray-300 text-sm mb-3">
                Experience the wonder of wildlife firsthand. Learn about our animals and conservation efforts on-site.
              </p>
              <Button variant="outline" size="sm" rightIcon={<ExternalLinkIcon className="w-4 h-4"/>} onClick={() => alert("Link to Zoo Website")}>
                Zoo Website
              </Button>
            </Card>
            <Card className="bg-earthy-brown/40">
              <LeafIcon className="w-10 h-10 text-accent-teal mb-3" />
              <h4 className="text-xl font-semibold text-sky-blue mb-2">Conservation Partners</h4>
              <p className="text-gray-300 text-sm mb-3">
                We collaborate with leading organizations to protect pygmy hippos and their habitats. Discover their work.
              </p>
              <Button variant="outline" size="sm" rightIcon={<ExternalLinkIcon className="w-4 h-4"/>} onClick={() => alert("Link to Conservation Partner")}>
                Partner Info
              </Button>
            </Card>
          </div>
          <div className="text-center mt-8">
            <h4 className="text-xl font-semibold text-sky-blue mb-2">Contact Us</h4>
            <p className="text-gray-300 mb-3">Have questions or want to collaborate? Reach out to our team.</p>
            <Button variant="secondary" onClick={() => window.location.href = 'mailto:info@hipposphere.ai'}>
              Email Us
            </Button>
          </div>
        </Section>

        <Section title="Our Sustainability Commitment" titleClassName="text-2xl font-semibold text-accent-teal mb-6">
          <Card className="bg-jungle-green/40">
             <LeafIcon className="w-10 h-10 text-accent-teal mb-3" />
            <p className="text-gray-300">
              HippoSphere AI and its host institution are dedicated to promoting sustainability. This includes employing eco-friendly practices within the habitat, educating the public on the importance of biodiversity and habitat preservation, and supporting research that contributes to a healthier planet for all species. We believe that technology and nature can thrive together, creating a more sustainable future.
            </p>
          </Card>
        </Section>
      </Section>
    </div>
  );
};